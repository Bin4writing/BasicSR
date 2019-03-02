
import os
import torch
import torch.nn as nn


class BaseModel():
    def __init__(self, opt):


    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    
    def generate(self):
        raise NotImplementedError('not a instance of generatable model')