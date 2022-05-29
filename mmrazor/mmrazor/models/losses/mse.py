# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module()
class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MSELoss, self).__init__()
        self.loss_weight = loss_weight
        self.mse = nn.MSELoss()

    def forward(self, y_s, y_t):
        if isinstance(y_s, (tuple, list)):
            assert len(y_s) == len(y_t)
            losses = []
            for s, t in zip(y_s, y_t):
                assert s.shape == t.shape
                losses.append(self.mse(s, t))
            loss = sum(losses)
        else:
            assert y_s.shape == y_t.shape
            loss = self.mse(y_s, y_t)
        return self.loss_weight * loss



