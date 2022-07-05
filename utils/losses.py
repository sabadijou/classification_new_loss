import torch.nn.functional as F
from torch import nn
import torch


class OurLoss(nn.Module):
    def __init__(self, gamma=2):
        super(OurLoss, self).__init__()
        self.gamma = gamma

    def nl(self, y_pred, target): return -y_pred[range(target.shape[0]), target].log().mean()

    def mse(self, y_pred, target): return (y_pred[range(target.shape[0]), target]**2).mean()

    def forward(self, y_pred1, y_true):

        # Find False Max
        y_pred = torch.nn.functional.softmax(y_pred1, dim=-1)
        tfs = -torch.argsort(-y_pred, axis=1) * -1
        ind = torch.where(tfs[:,0][:] == y_true[:])
        tfs[:,0][ind], tfs[:,1][ind] = tfs[:,1][ind], tfs[:,0][ind]
        target_f = tfs[:, 0]


        # calculate loss_1 and loss_2 ##################################
        loss_1 = self.nl(torch.clip(y_pred, 1e-5, 1-1e-5), y_true)
        loss_2 = self.nl(torch.clip(1 - y_pred, 1e-5, 1-1e-5), target_f)

        # calculate loss_2 coef ########################################
        beta = loss_1 / (loss_1 + loss_2)
        alpha = beta ** self.gamma

        # Final loss ###################################################
        loss = loss_1 + loss_2 * alpha

        return loss

class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma=2, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss


def choose_loss(loss_type='ourloss', gamma=2):
    if loss_type == 'ourloss':
        loss = OurLoss(gamma=gamma)
        return loss
    if loss_type == 'cross_entropy':
        loss = nn.CrossEntropyLoss()
        return loss
    if loss_type == 'focal_loss':
        loss = FocalLoss(gamma=gamma, reduction='mean')
        return loss

