from .backbones import ResNett
from  .heads import HeatHead
from .detectors import CtdetDetector, HeatMap

from .registry import BACKBONES, HEADS, DETECTORS
from .builder import build_backbone, build_head, build_detector