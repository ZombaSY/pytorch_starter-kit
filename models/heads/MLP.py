import torch.nn as nn

from models import utils


class SimpleClassifier(nn.Module):
    def __init__(self, in_features, hidden_dims, num_class, normalization=nn.BatchNorm1d, activation=nn.ReLU):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(in_features, hidden_dims),
            normalization(hidden_dims),
            activation(),

            nn.Linear(hidden_dims, num_class),
            # nn.Sigmoid()
        )

        self.apply(utils.init_weights)

    def forward(self, feat):

        return self.classifier(feat)
