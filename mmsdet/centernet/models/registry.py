from centernet.utils.registry import Registry


BACKBONES = Registry('backbone')
HEADS = Registry('head')
HEATMAP = Registry('heatmap')
DETECTORS = Registry('detector')

LOSSES = Registry('loss')
