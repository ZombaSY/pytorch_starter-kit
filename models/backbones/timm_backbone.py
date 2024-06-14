import timm
import torch.nn as nn
import torch


class BackboneLoader(nn.Module):
    def __init__(self, model_name, **kwconf):
        super().__init__()
        self.backbone = timm.create_model(model_name, **kwconf)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):

        return self.backbone.forward_features(x)

    def forward_perturb(self, x):
        x = x.clone().detach()
        x_perturb1 = self.dropout(x)
        x_perturb2 = x + (self.dropout(x * ((torch.rand(x.shape) - 0.5) * 2).cuda()) * 0.2)   # give random noise

        return [x_perturb1, x_perturb2]
