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
                    img_size).to(device)

    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path)
        
    else:
        # on cpu
        if device == "cpu":
            model.load_state_dict(torch.load(weights_path,
                                             map_location=torch.device('cpu')))
        # on gpu
        else:
            model.load_state_dict(torch.load(weights_path))
    
    return model.eval()


# Detecting and making tensor objects from each frame
def detect_image(model, img, tensor, img_size, conf_thres, nms_thres):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, conf_thres, nms_thres)
        
    return detections[0]
