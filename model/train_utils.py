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

def dense_lazy_couplet(out_features, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(out_features=out_features, bias=True),
        getattr(torch.nn, act_fun)(),
    )

def conv_couplet(in_channels, out_channels, act_fun, *args, **kwargs):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
        getattr(torch.nn, act_fun)(),
        torch.nn.MaxPool2d(kernel_size=(2, 2), ceil_mode=True),
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

def conv_block(in_channels, out_channels, act_fun, kernel_size):
    block = [
        conv_couplet(in_channels, out_channels, act_fun, kernel_size, padding="same")
        for in_channels, out_channels, act_fun, kernel_size in zip(
            [*in_channels],
            [*out_channels],
            [*act_fun],
            [*kernel_size],
        )
    ]
    return torch.nn.Sequential(*block)
    
class NeuralNetwork(BaseModel):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.pad_lons = torch.nn.CircularPad2d(config["circular_padding"])

        # CNN block
        self.conv_block = conv_block(
            [config["n_inputchannel"], *config["filters"][:-1]],
            [*config["filters"]],
            [*config["cnn_act"]],
            [*config["kernel_size"]],
        )

        # Flat layer
        self.flat = torch.nn.Flatten(start_dim=1)

        # Input Dense blocks
        self.denseblock = dense_block(
            config["hiddens_block"],
            config["hiddens_block_act"],
            #in_features=config["hiddens_block_in"],
        )

        # Final dense layer
        self.output = dense_couplet(
            out_features=config["hiddens_final"],
            # act_fun=config["hiddens_final_act"],
            in_features=config["hiddens_final_in"],
        )

    def forward(self,x):
        
        x = self.pad_lons(x)
        x = self.conv_block(x)
        x = self.flat(x)
        x = self.denseblock(x)
        x = self.output(x)
        
        return x




