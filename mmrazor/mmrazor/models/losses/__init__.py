# Copyright (c) OpenMMLab. All rights reserved.
from .cwd import ChannelWiseDivergence
from .kl_divergence import KLDivergence
from .weighted_soft_label_distillation import WSLD
from .soft_cross_entropy import SoftCrossEntropy
from .dist_kd import DIST
from .mse import MSELoss
from .maskd import MasKDLoss

__all__ = ['ChannelWiseDivergence', 'KLDivergence', 'WSLD', 'DIST', 'MSELoss', 'MasKDLoss', 'SoftCrossEntropy']
