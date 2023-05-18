from torch.nn import functional as F
import torch.nn as nn
import torch
import torchvision

from models import utils


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class Classifier_map_score_multi_stem(nn.Module):
    def __init__(self, dims=768, dim_inter=1024):
        super(Classifier_map_score_multi_stem, self).__init__()
        self.stem1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(dims, dim_inter),
            nn.BatchNorm1d(dim_inter),
            nn.ReLU(),
        )

        self.stem2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),

            nn.Linear(dims, dim_inter),
            nn.BatchNorm1d(dim_inter),
            nn.ReLU(),
        )

        self.classifier1 = nn.Sequential(
            nn.Linear(dim_inter + 1, dim_inter // 2),
            nn.BatchNorm1d(dim_inter // 2),
            nn.ReLU(),

            nn.Linear(dim_inter // 2, 1),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(dim_inter + 1, dim_inter // 2),
            nn.BatchNorm1d(dim_inter // 2),
            nn.ReLU(),

            nn.Linear(dim_inter // 2, 1),
        )

        self.apply(utils.init_weights)

    def forward(self, feat, score):
        feat1 = self.stem1(feat)
        feat2 = self.stem2(feat)

        score1 = self.classifier1(torch.cat([feat1, score[..., 0].unsqueeze(-1)], dim=1))
        score2 = self.classifier2(torch.cat([feat2, score[..., 1].unsqueeze(-1)], dim=1))
        score = torch.cat([score1, score2], dim=1)

        return score
