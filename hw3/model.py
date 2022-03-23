import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

class Model(nn.Module):
    num_classes = 11
    def __init__(self, name=None) -> None:
        super().__init__()
        if name == 'resnext':
            self.model = models.resnext101_32x8d(pretrained=False)
        elif name == 'effnet_b4':
            self.model = models.efficientnet_b4(pretrained=False)
        elif name == 'effnet_b5':
            self.model = models.efficientnet_b5(pretrained=False)
        elif name == 'convnext_small':
            self.model = models.convnext_small(pretrained=False)
        else:
            raise NotImplementedError
        # self.fc = nn.Sequential(nn.Linear(1000, 512),
        #                         nn.ReLU(),
        #                         nn.Linear(512, 128),
        #                         nn.ReLU(),

        #                         nn.Linear(128, self.num_classes)
        #                         )
        self.fc = nn.Sequential(nn.Linear(1000, 256),
                                nn.ReLU(),
                                nn.Linear(256, self.num_classes)
                                )
    def forward(self, x):
        return self.fc(self.model(x))


class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='sum',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def mixup(data, targets, alpha):
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets