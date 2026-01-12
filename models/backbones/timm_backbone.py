import timm
import torch.nn as nn


class BackboneLoader(nn.Module):
    def __init__(self, model_name, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(model_name, **kwargs)

    def forward(self, x):
        """
        To generate hierarchical feature maps
        we alternately use stage-side prediction rather than `forward_features()`
        """
        out = []

        x = self.backbone.stem(x)
        for stage in self.backbone.stages:
            x = stage(x)
            out.append(x)

        return out
