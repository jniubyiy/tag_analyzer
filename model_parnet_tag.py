# model_parnet_tag.py
import torch
import torch.nn as nn

class ParNetTag(nn.Module):
    """ Преобразует последовательность из 128 байтов в одноканальный парнет размерности 128.
        Параметры задаются через конструктор.
    """
    def __init__(self, hidden_dim=256, num_layers=4, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = 128
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 128
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, byte_sequence):
        return self.mlp(byte_sequence.float())