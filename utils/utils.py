"""Utility functions.

Edited by: Kirsten Mayer
Written by: Elizabeth A. Barnes

Functions
---------
prepare_device(device="gpu")
save_torch_model(model, filename)
load_torch_model(model, filename)
get_config(exp_name)
"""

import json
import torch
import numpy as np

def prepare_device(device="gpu"):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if device == "gpu":
        print('device available :', torch.cuda.is_available())
        print('device count: ', torch.cuda.device_count())
        print('current device: ',torch.cuda.current_device())
        print('device name: ',torch.cuda.get_device_name())
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device.type == 'cuda':
            print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        
        if not torch.cuda.is_available():
            print("Warning: Training will be performed on CPU.")
            
    elif device == "cpu":
        print("Training will be performed on CPU.")
        device = torch.device("cpu")
    else:
        raise NotImplementedError

    print('using device: ', device)
    return device


def save_torch_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_torch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


def get_config(exp_name):

    basename = "exp_"

    with open("config/config_" + exp_name[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert (
        config["exp_name"] == basename + exp_name[len(basename) :]
    ), "Exp_Name must be equal to config[exp_name]"

    return config


class MetricTracker:
    def __init__(self, *keys):

        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    def print(self, idx=None):
        for key in self.history.keys():
            if idx is None:
                print(f"  {key} = {self.history[key]:.5f}")
            else:
                print(f"  {key} = {self.history[key][idx]:.5f}")
