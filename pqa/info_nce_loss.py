import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class InfoNCELoss(nn.Module):
    def __init__(self, size_average=True):
        super(InfoNCELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target): # just to keep the api same

        batch_size = input.shape[0] # hardcode now
        new_probs = input.view(100, 10)
        indices = torch.tensor([0]).cuda()

        pos = torch.index_select(new_probs, 1, indices)
        sum = torch.sum(new_probs, dim=1).view(100, 1)
        e = pos/sum
        loss = torch.log(e)

        if self.size_average: return loss.mean()
        else: return loss.sum()
