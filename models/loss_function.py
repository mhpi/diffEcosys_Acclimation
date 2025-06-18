import torch
import torch.nn as nn
class RangeBoundLoss(nn.Module):
    # limit parameters from going out of range

    # Description: A Custom loss function that penalizes parameters going outside a specified lower and upper bound
    #
    # Inputs:
    # lb : Lower bounds for each parameter
    # ub : Upper bounds for each parameter

    def __init__(self, lb, ub):
        super(RangeBoundLoss, self).__init__()
        self.lb = torch.tensor(lb).cuda()
        self.ub = torch.tensor(ub).cuda()
    def forward(self, params, factor):

        # Compute the total loss based on by how much parameters exceed their allowed range

        # Inputs
        # params : List of parameter tensors to be constrained or penalized
        # factor : Scaling factor for the penalty term

        # outputs
        # loss : total loss for all parameters outside their specified bounds (< lb or > ub)

        factor = torch.tensor(factor).cuda()
        loss = 0
        for i in range(len(params)):
            lb = self.lb[i]
            ub = self.ub[i]
            upper_bound_loss = factor * torch.relu(params[i] - ub).mean()
            lower_bound_loss = factor * torch.relu(lb - params[i]).mean()
            loss = loss + upper_bound_loss + lower_bound_loss
        return loss

