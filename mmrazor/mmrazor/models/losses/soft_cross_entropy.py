# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class SoftCrossEntropy(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        use_sigmoid=False,
        loss_weight=1.0,
    ):
        super(SoftCrossEntropy, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight


    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if self.use_sigmoid:
            # bce loss
            preds_S = preds_S.sigmoid()
            preds_T = preds_T.sigmoid()
            pos_loss = -preds_T * (preds_S + 1e-8).log()
            neg_loss = -(1 - preds_T) * (1 - preds_S + 1e-8).log()
            loss = pos_loss + neg_loss
            loss = loss.mean()
            #loss = loss.view(loss.shape[0], -1).sum(1).mean()
            return loss
        if preds_S.ndim == 4:
            preds_S = preds_S.transpose(1, 3).reshape(-1, 80)
            preds_T = preds_T.transpose(1, 3).reshape(-1, 80)
        if self.use_sigmoid:
            preds_S = preds_S.sigmoid()
            preds_T = preds_T.sigmoid()
        else:
            assert len(preds_S.shape) == 2
            preds_S = preds_S.softmax(-1)
            preds_T = preds_T.softmax(-1)
        preds_T = preds_T.detach()
        loss = -torch.sum(preds_T * torch.log(preds_S + 1e-6)) / preds_S.shape[0]
        return self.loss_weight * loss
