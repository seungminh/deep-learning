#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:42:51 2019

@author: seungmin
"""


from area.area_designation import *
from readNet import *
from detectBox import *
from tracker.sort import *
from utils import *

import datetime
import argparse

import torch

from PIL import Image
import numpy as np

import cv2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-cfg", "--config_path", type=str, help="config file")
    
    parser.add_argument("-w", "--weights_path", type=str, help="weight file")
    
    parser.add_argument("-cls", "--class_path", type=str, help="class name file")
    
    parser.add_argument("-i", "--img_size", type=int, default=416, help="img_size")
    
    parser.add_argument("-conf", "--conf_thres", type=float, default=0.2, help="conf_thres")
    
    parser.add_argument("-nms", "--nms_thres", type=float, default=0.2, help="nms_thres")
    
    parser.add_argument("-vid", "--input_video", type=str, help="input video path")
    
    parser.add_argument("-sav", "--save_path", type=str, help="save path for trajectory")
    #parser.add_argument("--input_ip", type=str, help="input ip")
    opt = parser.parse_args()
    print(opt)



# 모델을 정의하기 위한 변수
yolomodel = {"config_path":opt.config_path,
             "weights_path":opt.weights_path,
             "class_path":opt.class_path,
             "img_size":opt.img_size,
             "conf_thres":opt.conf_thres,
             "nms_thres":opt.nms_thres
             }

# 디바이스 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
my_model = readNet(device, 
                   yolomodel["config_path"], 
                   yolomodel["img_size"], 
                   yolomodel["weights_path"])

# 객체명 리스트 불러오기
classes = utils.load_classes(yolomodel["class_path"])

# 텐서 타입 정의
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

# 경계박스 색깔
colors = np.random.randint(0, 255, size=(200, 3))

# 객체추적 알고리즘 초기화
mot_tracker = Sort()
memory = {}

# Draw area with the first frame of the video
vid = cv2.VideoCapture(opt.input_video)
ret, frame = vid.read()

#print("\n")
#print("Press 's' button to start design your area. Or 'p' button to pass this process.")

# Initializing area_Designation classs
CANVAS_SIZE = (vid.get(3),vid.get(4))
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

my_area = PolygonDrawer("Area", frame, CANVAS_SIZE, FINAL_LINE_COLOR, WORKING_LINE_COLOR) 
my_area.run()

# Counting objects by intersection 
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

readVid(opt.input_video,
        my_model,
        Tensor,
        yolomodel["img_size"],
        yolomodel["conf_thres"],
        yolomodel["nms_thres"],
        my_area,
        mot_tracker,
        colors,
        classes)