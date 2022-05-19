import torch
import torch.nn as nn


class DynamicWeightedLoss(nn.Module):
    def __init__(
        self,
        keys=["loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"],
    ):
        super(DynamicWeightedLoss, self).__init__()
        self.params = nn.ParameterDict(
            {k: nn.Parameter(torch.ones(1, requires_grad=True)) for k in keys}
        )

    def forward(self, x):
        loss_sum = 0
        for k in x.keys():
            loss_sum += 0.5 / (self.params[k] ** 2) * x[k] + torch.log(
                1 + self.params[k] ** 2
            )
        return loss_sum
