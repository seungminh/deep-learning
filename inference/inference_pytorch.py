#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:42:51 2019

@author: seungmin
"""


from models import *
from utils import *

import cv2
from tracker.sort import *

import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image
import numpy as np


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", type=str, default="config/yolov3-custom.cfg",
                        help="config file")
    
    parser.add_argument("--weights_path", type=str, default="weights/yolov3_ckpt_99_imagenet_470.pth",
                        help="weight file")
    
    parser.add_argument("--class_path", type=str, default="config/classes.names",
                        help="class name file")
    
    parser.add_argument("--img_size", type=int, default=416,
                        help="img_size")
    
    parser.add_argument("--conf_thres", type=float, default=0.9,
                        help="conf_thres")
    
    parser.add_argument("--nms_thres", type=float, default=0.2,
                        help="nms_thres")
    
    parser.add_argument("--input_video", type=str, default="input/pedestrian_06.mp4",
                        help="input video path")
    
    opt = parser.parse_args()
    print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = Darknet(opt.config_path, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    model.load_darknet_weights(opt.weights_path)
else:
    model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))
    
classes = utils.load_classes(opt.class_path)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def detect_image(img):
    # scale and pad image
    ratio = min(opt.img_size/img.size[0], opt.img_size/img.size[1])
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
    input_img = Variable(image_tensor.type(Tensor))
    
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
    return detections[0]


counter = 0
memory = {}

colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

mot_tracker = Sort()
vid = cv2.VideoCapture(opt.input_video)

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

#fourcc = cv2.VideoWriter_fourcc(*'MP4V')
ret,frame=vid.read()
#outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh),True)
#print(outvideo.isOpened())

frames = 0
frame_count = 0

# Starting object detection

(H, W) = (None, None)
writer = None

while(True):
    ret, frame = vid.read()
    if not ret:
        print("파일을 열 수 없습니다")
        break
    
    if W is None or H is None: (H, W) = frame.shape[:2]
    
    frames += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
    unpad_h = opt.img_size - pad_y
    unpad_w = opt.img_size - pad_x
    
    boxes = []
    confidences = []
    classIDs = []
    
    
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        box = detections[0:4]
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        
        boxes = []
        indexIDs = []
        previous = memory.copy()
        memory = {}
        
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            
            color = colors[int(cls_pred) % len(colors)]
            #color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            
            realtime = datetime.datetime.now().strftime("%H:%M:%S")
            
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
            cv2.rectangle(frame, (x1, y1-25), (x1+len(cls)*12, y1), color, -1)
            cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            print('{0} - {1}'.format(realtime, (cls, int(obj_id))))
                
            boxes.append([x1, y1, box_w, box_h])
            indexIDs.append(int(obj_id))
            memory[indexIDs[-1]] = boxes[-1]
        
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    
                    p0 = (int(x + (w)/2), int(y + (h)/2))
                    p1 = (int(x2 + (w2)/2), int(y2 + (h2)/2))
                    
                i +=1
    
    cv2.imshow('Stream', frame)
    
    ch = 0xFF & cv2.waitKey(100)
    if ch == 27:
        break
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30, (W, H), True)
    writer.write(frame)
    
writer.release()
vid.release()
cv2.destroyAllWindows()
