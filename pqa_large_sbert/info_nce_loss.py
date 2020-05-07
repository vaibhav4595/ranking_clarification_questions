import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pdb import set_trace as bp

class InfoNCELoss(nn.Module):

    def __init__(self, size_average=True):
        super(InfoNCELoss, self).__init__()
        self.size_average = size_average
        self.eps = 1e-12

    def forward(self, input, target): # just to keep the api same

        batch_size = input.shape[0] # hardcode now

        #print(batch_size)

        batch_size = batch_size//10

        input = torch.exp(input)

        new_probs = input.view(batch_size, 10)
        indices = torch.tensor([0]).cuda()

        pos = torch.index_select(new_probs, 1, indices)
        # Can be tried at the last layer as well -- check later
        # This line has to be removed
        # pos = F.relu(pos)

        #pos = torch.exp(pos)
        #sum = torch.sum(torch.exp(new_probs), dim=1).view(batch_size, 1)
        sum = torch.mean(new_probs, dim=1).view(batch_size, 1)

        e = (pos + self.eps)/(sum + self.eps)

        loss = -1 * torch.log(e + self.eps)

        if self.size_average: return loss.mean()
        else: return loss.sum()
