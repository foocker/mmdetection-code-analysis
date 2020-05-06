import torch
from torch import nn

from ..backbones import ResNett
from ..heads import HeatHead
from ..registry import HEATMAP
from ..builder import build_backbone, build_head


@HEATMAP.register_module
class HeatMap(nn.Module):
    def __init__(self, backbone, head, train_cfg=None, test_cfg=None, pretrained=None):
        super(HeatMap, self).__init__()
        self.backbone = build_backbone(backbone)
        self.heathead = build_head(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)


    def init_weights(self, pretrained=None):
        self.backbone.init_weights()
        self.heathead.init_weights()

    def forward(self, x):
        resx = self.backbone(x)
        heatmap = self.heathead(resx)

        return heatmap