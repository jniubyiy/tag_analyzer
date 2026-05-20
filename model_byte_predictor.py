# model_byte_predictor.py
"""
Модель предсказания байтов тега на основе парнета от ParNetTag.
Входы:
- parnet_vector: [B, cond_dim] — выход ParNetTag (одномерный парнет)
- byte_sequence: [B, byte_len] — префикс (реальные байты, остальные 0)
Выход:
- probs: [B, byte_len, 256] — распределение вероятностей по 256 байтам (сумма по последнему измерению = 1)
Архитектура: конкатенация входов → MLP → softmax.
"""

import torch
import torch.nn as nn


class BytePredictor(nn.Module):
    def __init__(self, cond_dim=128, byte_len=128, hidden_dims=None, dropout=0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        input_dim = cond_dim + byte_len  # 256
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        # Выходной слой: предсказываем для каждой позиции 256 логитов, затем softmax
        layers.append(nn.Linear(in_dim, byte_len * 256))
        self.net = nn.Sequential(*layers)
        self.byte_len = byte_len
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, parnet_vector: torch.Tensor, byte_sequence: torch.Tensor):
        """
        Args:
            parnet_vector: [B, cond_dim]
            byte_sequence: [B, byte_len]
        Returns:
            probs: [B, byte_len, 256] – вероятности, сумма по последнему измерению = 1
        """
        x = torch.cat([parnet_vector, byte_sequence], dim=-1)  # [B, cond_dim+byte_len]
        logits = self.net(x)                                   # [B, byte_len * 256]
        logits = logits.view(-1, self.byte_len, 256)           # [B, byte_len, 256]
        probs = self.softmax(logits)                           # [B, byte_len, 256]
        return probs