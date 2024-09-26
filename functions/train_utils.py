'''Network modules for pytorch models. 

Edited by: Kirsten Mayer
Written by: Elizabeth A. Barnes

Functions
---------
dense_couplet(in_features, out_features, act_fun, *args, **kwargs)
dense_lazy_couplet(out_features, act_fun, *args, **kwargs) #assumes in_features
dense_block(out_features, act_fun)

Classes
---------
TorchModel()

'''

import numpy as np
import torch


def dense_couplet(in_features, out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )
    
def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )

def dense_block(out_features, act_fun, in_features=None):
    if in_features is None:
        block = [
            dense_lazy_couplet(out_channels, act_fun)
            for out_channels, act_fun in zip([*out_features], [*act_fun])
        ]
        return torch.nn.Sequential(*block)
    else:
        block = [
            dense_couplet(in_features, out_features, act_fun)
            for in_features, out_features, act_fun in zip(
                [*in_features], [*out_features], [*act_fun]
            )
        ]
        return torch.nn.Sequential(*block)
    
class NeuralNetwork(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Input Dense blocks
        self.denseblock = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            in_features=config["hiddens_block_in"],
        )

        # Final dense layer
        self.finaldense = dense_couplet(
            out_features=config["hiddens_final"],
            act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )

        # Output layer
        self.output = torch.nn.Linear(
            in_features=config["hiddens_final"], out_features=1, bias=True
        )

    def forward(self,x):

        x = self.denseblock(x)
        x = self.finaldense(x)
        x = self.output(x)
        
        return x


def train(dataloader, model, loss_fn, optimizer,scheduler):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error/loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 5 == 0:
            loss, current = loss.iten(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    scheduler.step()


def val(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # remove batch normalization, dropout, etc.
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Vall Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")

    if epoch == 0:
        print('...saved first epoch...')
        best_mod = copy.deepcopy(model)
        best_skill = copy.deepcopy(val_loss)
    elif val_loss < best_skill:
        print('...saving...')
        best_mod = copy.deepcopy(model)
        best_skill = copy.deepcopy(val_loss)
    else:
        print('No improvement in validation performance')



