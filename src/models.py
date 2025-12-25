# src/models.py
import torch
import torch.nn as nn

class FrameEncoder(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B*T, D)
        h = self.net(x)
        return h.view(B, T, -1)

class SeqModel(nn.Module):
    def __init__(self, input_dim, frame_hidden=128, lstm_hidden=256, num_layers=2, num_classes=10):
        super().__init__()
        self.encoder = FrameEncoder(input_dim, hidden=frame_hidden)
        self.lstm = nn.LSTM(frame_hidden, lstm_hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden*2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        enc = self.encoder(x)
        out, _ = self.lstm(enc)
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
