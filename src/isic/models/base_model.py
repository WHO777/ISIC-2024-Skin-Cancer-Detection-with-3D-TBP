from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.isic.models.timm_model import TimmBackbone


class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: int):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        x_avg = F.adaptive_avg_pool2d(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, self.output_size)
        return 0.5 * (x_avg + x_max)


class Head(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_features: int,
        embedding_size: Optional[int] = None,
        dropout_rate: Optional[float] = None,
    ):
        super(Head, self).__init__()
        self.drop = nn.Dropout(dropout_rate) if dropout_rate is not None else None
        self.embedding = (
            nn.Linear(input_features, embedding_size)
            if embedding_size is not None
            else None
        )
        self.fc = nn.Linear(
            embedding_size if embedding_size is not None else input_features,
            num_classes,
        )

    def forward(self, x):
        if self.drop is not None:
            x = self.drop(x)
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.fc(x)
        return x


class Model(torch.nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        embedding_size: Optional[int] = None,
        dropout_rate: Optional[int] = None,
        activation: Optional[str] = "sigmoid",
    ):
        super(Model, self).__init__()
        self._backbone = TimmBackbone(backbone_name, pretrained)
        self._pool = AdaptiveAvgMaxPool2d(1)
        self._head = Head(
            num_classes, self._backbone.num_features, embedding_size, dropout_rate
        )
        activation_fn = torch.nn.Identity
        if activation == "sigmoid":
            activation_fn = torch.sigmoid
        elif activation == "softmax":
            activation_fn = torch.softmax
        self._activation = activation_fn

    def forward(self, images):
        x = self._backbone(images)
        x = self._pool(x).view(-1, x.shape[1])
        x = self._head(x)
        x = self._activation(x)
        return x
