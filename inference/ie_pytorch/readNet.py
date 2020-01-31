#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:13:24 2020

@author: seungmin
"""


from models.models import *
from utils import *

import torch
from torchvision import transforms
from torch.autograd import Variable



def readNet(device, config_path, img_size, weights_path):
    
    # Load model and weights
    model = Darknet(config_path, 
                    img_size).to(device, non_blocking = True)

    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path)
        
    #else:
        #model.load_state_dict(torch.load(weights_path,
        #                                 map_location=torch.device('cpu')))
        
        # on cpu
    elif device == "cpu":
        model.load_state_dict(torch.load(weights_path,
                                         map_location=torch.device('cpu')))
        # on gpu
    elif device == "cuda":
        model.load_state_dict(torch.load(weights_path))
    
    
    return model.eval()


