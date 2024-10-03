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
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("Warning: MPS device not found." "Training will be performed on CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        device = torch.device("cpu")
    else:
        raise NotImplementedError

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
