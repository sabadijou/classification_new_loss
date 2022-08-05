import torch.nn.functional as F
from utils.gpu import device
from torch import nn
import torch


class OurLoss(nn.Module):
    def __init__(self, gamma=2):
        super(OurLoss, self).__init__()
        self.gamma = gamma

    def nl(self, y_pred, target): return -y_pred[range(target.shape[0]), target].log().mean()

    def mse(self, y_pred, target): return (y_pred[range(target.shape[0]), target]**2).mean()

    def forward(self, y_pred1, y_true):
        # print(y_true)
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


class ourLoss_weighted(nn.Module):
    def __init__(self, gamma=2):
        super (ourLoss_weighted, self).__init__ ()
        self.gamma = gamma
    # def softmax(self, x): return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
    def nl(self, y_pred, target): return -y_pred[range (target.shape[0]), target].log ().mean ()

    def nl_sep(self, y_pred, target): return -y_pred[range (target.shape[0]), target].log ()

    def mse(self, y_pred, target): return (y_pred[range (target.shape[0]), target] ** 2).mean ()

    def Bhattacharyya_dist(self, y_pred, target): return -(sum (y_pred[range (target.shape[0]), target] ** (.5))).log ()

    def forward(self, y_pred1, y_true, weights_tensor):
        # y_pred = self.softmax(y_pred1)
        y_pred = torch.nn.functional.softmax (y_pred1, dim=-1)
        # y_pred = torch.nan_to_num(y_pred, nan=0.0, posinf=1.0)
        tfs = -torch.argsort (-y_pred, axis=1) * -1
        ind = torch.where (tfs[:, 0][:] == y_true[:])
        tfs[:, 0][ind], tfs[:, 1][ind] = tfs[:, 1][ind], tfs[:, 0][ind]
        target_f = tfs[:, 0]  # Max falses
        ########################################################################
        ########################################################################  MSE + MSE
        '''loss_1 = self.mse(1-y_pred, y_true)
        loss_2 = self.mse(y_pred, target_f)
        loss = (loss_1 + loss_2)'''

        ########################################################################  CE + alpha*CE
        '''loss_1 = self.nl(torch.clip(y_pred, 1e-5, 1-1e-5), y_true)
        loss_2 = self.nl(torch.clip(1 - y_pred, 1e-5, 1-1e-5), target_f)
        beta = loss_1 / (loss_1 + loss_2)

        alpha = beta ** self.gamma
        # alpha = torch.exp(beta)
        # print('self.gm', self.gamma)
        loss = loss_1 + loss_2 * alpha'''

        ########################################################################  w1*CE + w2*CE
        loss_1 = self.nl_sep (torch.clip (y_pred, 1e-5, 1 - 1e-5), y_true)
        loss_2 = self.nl_sep (torch.clip (1 - y_pred, 1e-5, 1 - 1e-5), target_f)

        w1 = (weights_tensor[range (y_true.shape[0]), y_true]).to(device)
        w2 = (weights_tensor[range (target_f.shape[0]), target_f]).to(device)

        loss = (w1 * loss_1).mean () + (w2 * loss_2).mean ()

        ########################################################################  CE + Focal
        '''loss_1 = self.nl(y_pred, y_true)
        loss_2 = self.nl(torch.clamp(1 - y_pred, 1e-7, 1-1e-7), target_f)
        pt = torch.exp(-loss_1)
        loss_2 = ((1 - pt) ** self.gamma * loss_2).mean()
        loss = loss_1 + loss_2 '''

        ########################################################################  Bhattacharyya_dist
        '''loss_1 = self.Bhattacharyya_dist(torch.clamp(y_pred, 1e-7, 1-1e-7), y_true)
        loss_2 = self.Bhattacharyya_dist(torch.clamp(1 - y_pred, 1e-7, 1-1e-7), target_f)
        loss = loss_1 + loss_2'''

        ######################################################################## CE all class
        '''y_pred = torch.clip(y_pred, 1e-5, 1-1e-5)
        # alpha = 1
        CE_all_neg = torch.sum(-(1-y_pred).log(),1)
        CE_true = -y_pred[range(y_true.shape[0]), y_true].log()
        CE_true_neg = -(1-y_pred[range(y_true.shape[0]), y_true]).log()
        loss_sum = (1/self.gamma)*(CE_all_neg - CE_true_neg) + CE_true
        loss = loss_sum.sum()
        # print(loss)'''

        ########################################################################

        ########################################################################
        ########################################################################
        if torch.isnan (loss):
            print (loss)

        return loss

def choose_loss(loss_type='ourloss', gamma=2):
    if loss_type == 'ourloss':
        loss = OurLoss(gamma=gamma)
        return loss
    if loss_type == 'cross_entropy':
        loss = nn.CrossEntropyLoss()
        return loss
    if loss_type == 'ourLoss_weighted':
        loss = ourLoss_weighted(gamma=2)
        return loss

    if loss_type == 'focal_loss':
        loss = FocalLoss(gamma=gamma, reduction='mean')
        return loss


