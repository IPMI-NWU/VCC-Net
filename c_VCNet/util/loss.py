import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1e-6
        dice = 0.
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                    pred[:, i].sum(dim=1).sum(dim=1) +
                    target[:, i].sum(dim=1).sum(dim=1) + smooth)
        dice = dice / pred.size(1)
        # loss_fun = nn.BCELoss()
        # BCE = loss_fun(pred.to(torch.float32), target.to(torch.float32))
        return torch.clamp((1 - dice).mean(), 0, 1)
