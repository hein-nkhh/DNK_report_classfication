import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (batch_size, seq_len, hidden_size)
        weights = self.attn(x).squeeze(-1)  # (batch_size, seq_len)
        weights = torch.softmax(weights, dim=1)
        output = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, hidden_size)
        return output

class OpinionBERTWithContrastive(nn.Module):
    def __init__(self, model_name, num_classes, embedding_dim=256, freeze_layers=2, dropout_rate=0.3, msd_samples=5):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_size = self.bert.config.hidden_size

        # CNN branch với multi-kernel
        self.convs = nn.ModuleList([
            nn.Conv1d(self.hidden_size, 100, kernel_size=k, padding=k//2) for k in [3, 4, 5]
        ])
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # BiGRU + Attention pooling
        self.bigru = nn.GRU(self.hidden_size, 256, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPooling(512)

        # Dropout + LayerNorm
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(100 * len(self.convs) + 512)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(100 * len(self.convs) + 512, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        # Multi-Sample Dropout setup
        self.msd_samples = msd_samples
        self.classifier = nn.Linear(100 * len(self.convs) + 512, num_classes)
        self.dropout_list = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(msd_samples)])

        # Freeze embedding + layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(freeze_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        x_cnn = last_hidden.transpose(1, 2)
        cnn_outputs = [self.global_max_pool(F.relu(conv(x_cnn))).squeeze(-1) for conv in self.convs]
        cnn_out = torch.cat(cnn_outputs, dim=1)

        gru_out, _ = self.bigru(last_hidden)
        attn_out = self.attn_pool(gru_out)

        combined = torch.cat((cnn_out, attn_out), dim=1)
        combined = self.layer_norm(combined)

        # Projection head (contrastive)
        embedding = self.projection(combined)
        embedding = F.normalize(embedding, p=2, dim=1)

        # Multi-Sample Dropout for classification logits
        logits_sum = 0
        for i in range(self.msd_samples):
            dropped = self.dropout_list[i](combined)
            logits_sum += self.classifier(dropped)
        logits = logits_sum / self.msd_samples

        return embedding, logits