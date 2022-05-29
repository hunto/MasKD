import torch
import torch.nn as nn

from ..builder import LOSSES


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


@LOSSES.register_module()
class DIST(nn.Module):
    def __init__(self, beta, gamma, use_sigmoid=False, loss_weight=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        self.loss_weight = loss_weight

    def forward(self, y_s, y_t):
        assert y_s.ndim in (2, 4)
        if y_s.ndim == 4:
            num_classes = y_s.shape[1]
            y_s = y_s.transpose(1, 3).reshape(-1, num_classes)
            y_t = y_t.transpose(1, 3).reshape(-1, num_classes)
        if self.use_sigmoid:
            y_s = y_s.sigmoid()
            y_t = y_t.sigmoid()
        else:
            y_s = y_s.softmax(dim=1)
            y_t = y_t.softmax(dim=1)

        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        return self.loss_weight * kd_loss



