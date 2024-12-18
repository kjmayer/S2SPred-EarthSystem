import torch
import numpy as np

class weighted_l1_loss(torch.nn.Module):
    def __init__(self, weights):
        super(weighted_l1_loss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        # Calculate weighted L1 loss
        # weighted_loss = torch.abs(output - target) * self.weights
        norm_weights = self.weights #/torch.mean(self.weights)
        weighted_loss = torch.abs(output - target) * norm_weights
        return torch.mean(weighted_loss)

def weighted_l1_loss_tf(target, output, weights):
    norm_weights = weights#/np.mean(self.weights)
    weighted_loss = np.abs(output - target) * norm_weights
    return np.mean(weighted_loss)

class weighted_acc_loss(torch.nn.Module):
    def __init__(self,weights):
        super(weighted_acc_loss,self).__init__()
        self.weights = weights
    def forward(self, output, target):
        w = self.weights/torch.mean(self.weights)
        num = torch.sum(w * output * target)
        denom = torch.sqrt(torch.sum(w * output ** 2) * torch.sum(w * target ** 2))
        acc = num/denom
        return acc

# Functions to evaluate the skill of model while training
# have to send to cpu and convert to numpy for code in base_trainer.py
def ACC(output, target):
    with torch.no_grad():
        num = torch.sum(output * target)
        denom = torch.sqrt(torch.sum(output ** 2) * torch.sum(target ** 2))
        acc = num/denom
        return acc.cpu().numpy()


def MAE(output, target):
    with torch.no_grad():
        loss = torch.abs(output-target)
        return torch.mean(loss).cpu().numpy()


def MSE(output, target):
    with torch.no_grad():
        loss = (output - target)**2
        return torch.mean(loss).cpu().numpy()
