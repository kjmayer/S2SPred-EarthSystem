import torch

class weighted_l1_loss(torch.nn.Module):
    def __init__(self, weights):
        super(weighted_l1_loss, self).__init__()
        self.weights = weights

    def forward(self, output, target):
        # Calculate weighted L1 loss
        weighted_loss = torch.abs(output - target) * self.weights
        return torch.mean(weighted_loss)


# Functions to evaluate the skill of model while training
# have to send to cpu and convert to numpy for code in base_trainer.py
def MAE(output,target):
    with torch.no_grad():
        loss = torch.abs(output-target)
        return torch.mean(loss).cpu().numpy()

def MSE(output,target):
    with torch.no_grad():
        loss = (output - target)**2
        return torch.mean(loss).cpu().numpy()