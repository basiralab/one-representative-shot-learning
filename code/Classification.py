# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 01:35:30 2020

@author: Mohammed Amine
"""

import pickle
import numpy as np
import torch
import random

import main_gcn_dgn as main_gcn

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

dataset = "example"
view = 0
test_accs = main_gcn.test_scores(dataset, view)
