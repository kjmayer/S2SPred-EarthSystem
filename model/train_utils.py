'''Network modules for pytorch models. 

Edited by: Kirsten Mayer
Written by: Elizabeth A. Barnes

Functions
---------
dense_couplet(in_features, out_features, act_fun, *args, **kwargs)
dense_block(out_features, act_fun)

Classes
---------
NeuralNetwork()

'''

import numpy as np
import torch
from base.base_model import BaseModel

def dense_couplet(in_features, out_features, act_fun=False, *args, **kwargs):
    if not act_fun:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True))
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=out_features, bias=True),
            getattr(torch.nn, act_fun)(),
        )

def dense_block(out_features, act_fun, in_features):
    block = [
        dense_couplet(in_features, out_features, act_fun)
        for in_features, out_features, act_fun in zip(
            [*in_features], [*out_features], [*act_fun]
        )
    ]
    return torch.nn.Sequential(*block)

class NeuralNetwork(BaseModel):

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
        self.output = dense_couplet(
            out_features=config["hiddens_final"],
            # act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )

    def forward(self,x):

        x = self.denseblock(x)
        x = self.output(x)
        
        return x




