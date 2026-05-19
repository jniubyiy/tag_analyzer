# model_parnet_tag.py
import torch
import torch.nn as nn
import config_training_tag_unit as cfg

class ParNetTag(nn.Module):
    """
    Преобразует последовательность из 128 байтов в одноканальный парнет размерности 128.
    Использует dropout из конфига.
    """
    def __init__(self, hidden_dim=None, num_layers=None):
        if hidden_dim is None:
            hidden_dim = cfg.TAG_MODEL_HIDDEN_DIM
        if num_layers is None:
            num_layers = cfg.TAG_MODEL_NUM_LAYERS
        super().__init__()
        layers = []
        in_dim = 128
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 128
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                if cfg.TAG_MODEL_DROPOUT > 0:
                    layers.append(nn.Dropout(cfg.TAG_MODEL_DROPOUT))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, byte_sequence):
        return self.mlp(byte_sequence.float())