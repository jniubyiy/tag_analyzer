# model_conditioned_parnet_classifier.py
"""
Модель классификатора по спецификации пользователя.

Архитектура (24 слоя):
1.  Свертка 4->8 каналов, stride=2 (256->128)               # [B,8,128,128]
2.  Обогащение conditioning вектором (размер 128)
3.  Линейное сжатие conditioning: 128 -> 64
4.  Свертка 8->8, stride=2 (128->64)                        # [B,8,64,64]
5.  Обогащение conditioning вектором (размер 64)
6.  Линейное сжатие conditioning: 64 -> 32
7.  Свертка 8->8, stride=2 (64->32)                         # [B,8,32,32]
8.  Обогащение conditioning вектором (размер 32)
9.  Линейное сжатие conditioning: 32 -> 16
10. Свертка 8->8, stride=2 (32->16)                         # [B,8,16,16]
11. Обогащение conditioning вектором (размер 16)
12. Линейное сжатие conditioning: 16 -> 8
13. Свертка 8->8, stride=2 (16->8)                          # [B,8,8,8]
14. Обогащение conditioning вектором (размер 8)
15. Линейное сжатие conditioning: 8 -> 4
16. Свертка 8->8, stride=2 (8->4)                           # [B,8,4,4]
17. Обогащение conditioning вектором (размер 4)
18. Flatten: [B,8,4,4] -> [B,128] (8*4*4=128)
19. Линейный слой 128 -> 64
20. Линейный слой 64 -> 32
21. Линейный слой 32 -> 16
22. Линейный слой 16 -> 8
23. Линейный слой 8 -> 4
24. Линейный слой 4 -> 1 (логит)

Примечание: Conditioning вектор изначально имеет размер cond_dim=128.
На каждом этапе обогащения он сначала проецируется на нужную размерность,
а затем используется как gamma и beta для FiLM-модуляции (гамма * x + бета).

Все свёртки: kernel_size=3, stride=2, padding=1 (сохраняют размерность после stride).
"""

import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """FiLM: gamma * x + beta, gamma,beta: [B, C] -> [B, C, 1, 1]."""
    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)   # [B,C,1,1]
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


class ConditionedParNetClassifier(nn.Module):
    def __init__(self, compressed_channels=4, cond_dim=128, dropout=0.0):
        super().__init__()
        # Конфигурация сокращений conditioning
        self.cond_dims = [128, 64, 32, 16, 8, 4]   # для слоёв 2,5,8,11,14,17
        # Количество свёрточных блоков (6 штук, каждый уменьшает разрешение в 2 раза)
        # Размеры: 256→128→64→32→16→8→4
        self.num_blocks = 6
        base_channels = 8   # фиксированное число каналов после первого слоя

        # ---------- Слои 1,4,7,10,13,16 : свёртки с stride=2 ----------
        self.convs = nn.ModuleList()
        in_ch = compressed_channels
        for i in range(self.num_blocks):
            conv = nn.Conv2d(in_ch, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
            bn = nn.BatchNorm2d(base_channels)
            relu = nn.ReLU(inplace=True)
            self.convs.append(nn.Sequential(conv, bn, relu))
            in_ch = base_channels   # после первого блока каналов всегда 8

        # ---------- MLP для генерации параметров FiLM (обогащение) ----------
        # Для каждого из 6 блоков нужны gamma и beta размерности base_channels (8)
        # Перед каждым FiLM conditioning вектор проецируется на соответствующую размерность
        self.film_mlps = nn.ModuleList()
        # также для каждого блока нужны линейные слои для сжатия conditioning (слои 3,6,9,12,15)
        self.cond_reducers = nn.ModuleList()

        # conditioning_dim на входе (cond_dim) = 128
        curr_cond_dim = cond_dim
        for i, target_dim in enumerate(self.cond_dims):
            # MLP для генерации gamma,beta из conditioning (после проецирования на target_dim)
            # Выход: 2 * base_channels (gamma + beta)
            mlp = nn.Sequential(
                nn.Linear(target_dim, target_dim),
                nn.ReLU(inplace=True),
                nn.Linear(target_dim, 2 * base_channels)
            )
            self.film_mlps.append(mlp)

            # Редукция conditioning для следующего этапа (кроме последнего)
            if i < len(self.cond_dims) - 1:
                reducer = nn.Linear(curr_cond_dim, self.cond_dims[i+1])
                self.cond_reducers.append(reducer)
                curr_cond_dim = self.cond_dims[i+1]

        # Слой 18: Flatten -> [B, 8*4*4] = [B,128]
        self.flatten = nn.Flatten()

        # Слои 19-24: линейная часть классификатора
        self.fc = nn.Sequential(
            nn.Linear(8 * 4 * 4, 64),   # 128 -> 64
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, compressed_parnet: torch.Tensor, conditioning_vector: torch.Tensor):
        """
        compressed_parnet: [B, 4, 256, 256] (ожидается разрешение 256)
        conditioning_vector: [B, 128] - выход ParNetTag
        """
        x = compressed_parnet
        cond = conditioning_vector   # [B,128]

        for i in range(self.num_blocks):
            # Свёрточный блок (слой 1,4,7,10,13,16)
            x = self.convs[i](x)   # [B,8, H/2, W/2]

            # Обогащение FiLM (слои 2,5,8,11,14,17)
            # Сначала проецируем conditioning на нужную размерность
            target_dim = self.cond_dims[i]   # 128,64,32,16,8,4
            # Для первого блока cond уже 128, для следующих будем редуцировать
            # Но cond текущий может быть больше target_dim, поэтому проецируем через временный слой?
            # У нас уже есть для каждого блота свой mlp, принимающий conditioning нужной размерности.
            # Для этого нам нужно привести cond к target_dim.
            if i == 0:
                cond_proj = cond   # уже 128
            else:
                # Используем редуктор для предыдущего шага (он уже уменьшил cond до нужного)
                # cond уже имеет размер target_dim после редукции
                cond_proj = cond
            # Генерируем gamma, beta
            params = self.film_mlps[i](cond_proj)          # [B, 2*base_channels]
            gamma = params[:, :8]   # 8 каналов
            beta  = params[:, 8:]
            # Применяем FiLM
            x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)

            # Редукция conditioning для следующего шага (кроме последнего)
            if i < len(self.cond_reducers):
                cond = self.cond_reducers[i](cond)   # уменьшаем размерность: 128->64->32->16->8->4

        # После всех блоков x: [B,8,4,4]
        x = self.flatten(x)          # [B,128]
        logit = self.fc(x)           # [B,1]
        return logit