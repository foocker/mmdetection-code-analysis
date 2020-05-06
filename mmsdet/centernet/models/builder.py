from torch import nn 
from centernet.utils import build_from_cfg

from .registry import (BACKBONES, HEADS, DETECTORS, HEATMAP)


def build(cfg, registry, default_args=None):
    return build_from_cfg(cfg, registry, default_args)

def build_backbone(cfg):
    return build(cfg, BACKBONES)

def build_head(cfg):
    return build(cfg, HEADS)

def build_heatmap(cfg):
    return build(cfg, HEATMAP)

def build_detector(cfg):
    return build(cfg, DETECTORS)    # here can remove train/test_cfg