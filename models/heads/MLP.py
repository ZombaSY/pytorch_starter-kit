import torch
import torch.nn as nn

from models import utils
from timm.models.layers import DropPath


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, num_class, normalization='BatchNorm1d', activation='ReLU', dropblock=True):
        super().__init__()

        normalization = getattr(nn, normalization)
        activation = getattr(nn, activation)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            activation(),
            DropPath(0.2) if dropblock else nn.Identity(),
            normalization(in_features),
            nn.Linear(in_features, num_class),
        )

        self.apply(utils.init_weights)

    def forward(self, feat):

        return self.classifier(feat)


class SimpleClassifierTransformer(nn.Module):
    def __init__(self, in_features, num_class, normalization='BatchNorm1d', activation='ReLU', dropblock=True):
        super().__init__()

        normalization = getattr(nn, normalization)
        activation = getattr(nn, activation)

        self.classifier = nn.Sequential(
            activation(),
            DropPath(0.1) if dropblock else nn.Identity(),
            normalization(in_features),

            nn.Linear(in_features, num_class),
        )

        self.apply(utils.init_weights)

    def forward(self, feat):
        return self.classifier(feat[:, 0])  # use class token as input
