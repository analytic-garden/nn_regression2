#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Net.py - a simple neural net class for regression
author: Bill Thompson
license: GPL 3
copyright: 2022-04-04
"""

import torch

# a sumple NN class
class RNet(torch.nn.Module):    
    def __init__(self, 
                 hidden_size = 10, 
                 num_layers = 1):
        super(RNet, self).__init__()
        self.rnn1 = torch.nn.RNN(input_size = 1, 
                                 hidden_size = hidden_size,
                                 num_layers = num_layers)
        self.linear1 = torch.nn.Linear(hidden_size, 2)
        self.leaky_rlu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = x.unsqueeze(dim = 1)
        x, _ = self.rnn1(x)
        x = self.leaky_rlu(x)
        x = self.linear1(x)

        return x.squeeze(dim = 1) # RNN
