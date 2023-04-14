import math
import torch
import torch.nn as nn


def dice_coeff(inputs):
    # inputs: [B, T, H*W]
    pred = inputs[:, None, :, :]
    target = inputs[:, :, None, :]

    mask = pred.new_ones(pred.size(0), target.size(1), pred.size(2))
    mask[:, torch.arange(mask.size(1)), torch.arange(mask.size(2))] = 0

    a = torch.sum(pred * target, -1)
    b = torch.sum(pred * pred, -1) + 1e-12
    c = torch.sum(target * target, -1) + 1e-12
    d = (2 * a) / (b + c)
    d = (d * mask).sum() / mask.sum()
    return d


class MaskModule(nn.Module):

    def __init__(self, channels, num_tokens=8, use_prob=False):
        super().__init__()
        self.mask_token = nn.Parameter(torch.randn(num_tokens, channels).normal_(0, 0.01))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # fan-out
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_mask(self, x):
        N, C, H, W = x.shape
        mask_token = self.mask_token.expand(N, -1, -1)  # [N, T, C]
        k = x.view(N, -1, H * W)
        attn = mask_token @ k  # [N, T, H * W]
        attn = attn.sigmoid()
        attn = attn.view(N, -1, H, W)
        return attn

    def forward_train(self, x):
        mask = self.forward_mask(x)
        out = x.unsqueeze(1) * mask.unsqueeze(2)  # [N, T, C, H, W]
        out = out.sum(1)
        # loss
        mask_loss = dice_coeff(mask.flatten(2))
        return out, mask_loss

    def forward(self, x):
        return self.forward_train(x)


class MasKDLoss(nn.Module):

    def __init__(self, channels, num_tokens=8, use_prob=False,  weight_s_mask_warmup=200000000, pretrained=''):
        super().__init__()
        self.weight_s_mask_warmup = weight_s_mask_warmup
        self.mask_module = MaskModule(channels, num_tokens, use_prob)
        ckpt = torch.load(pretrained, map_location='cpu')
        self.mask_module.load_state_dict(ckpt, strict=True)

    def forward(self, y_s, y_t):
        mask = self.mask_module.forward_mask(y_t)
        masked_y_s = y_s.unsqueeze(1) * \
            mask.unsqueeze(2)  # [N, n_masks, C, H, W]
        masked_y_t = y_t.unsqueeze(1) * \
            mask.unsqueeze(2)  # [N, n_masks, C, H, W]

        square_loss = (masked_y_s - masked_y_t)**2
        square_loss = square_loss.sum((3, 4))  # [N, n_masks, C]
        square_loss = square_loss / (mask.sum((2, 3)).unsqueeze(-1) + 1e-5)
        square_loss = square_loss.mean()
        loss = square_loss * 0.5
        return loss

