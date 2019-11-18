#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:42:51 2019

@author: seungmin
"""


from models import *
from utils import *

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
    parser.add_argument("--config_path", type=str, default="config/yolov3-custom.cfg", help="config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="weight file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="class name file")
    parser.add_argument("--img_size", type=int, default=416, help="img_size")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="conf_thres")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="nms_thres")
    parser.add_argument("--input_video", type=str, help="input video path")
    #parser.add_argument("--input_ip", type=str, help="input ip")
    opt = parser.parse_args()
    print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and weights
model = Darknet(opt.config_path, img_size=opt.img_size).to(device)
model.load_state_dict(torch.load(opt.weights_path, map_location=torch.device('cpu')))
model.eval()
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

#videopath = 'http://192.168.40.187:8080/video'
videopath = opt.input_video


import cv2
from sort import *

line_01 = [(265,330), (280,450)]
line_02 = [(435,310), (500,420)]

counter = 0
memory = {}

#np.random.seed(42)
#colors = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
colors=[(255,0,0),(0,255,0),(0,0,255),(255,0,255),(128,0,0),(0,128,0),(0,0,128),(128,0,128),(128,128,0),(0,128,128)]

vid = cv2.VideoCapture(videopath)
mot_tracker = Sort()

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]
print ("Video size", vw,vh)
outvideo = cv2.VideoWriter(videopath.replace(".mp4", "-det.mp4"),fourcc,20.0,(vw,vh),True)
print(outvideo.isOpened())

def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

frames = 0
frame_count = 0

while(True):
    ret, frame = vid.read()
    if not ret:
        break
    
    
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
    
    # draw semi transparent polygon
    overlay = frame.copy()
    pts = np.array([[265,330],[280,450],[500,420],[435,310]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(frame, [pts], (0,255,0))
    opacity = 0.7
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    
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
            
            color = colors[int(obj_id) % len(colors)]
            #color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            
            realtime = datetime.datetime.now().strftime("%H:%M:%S")
            
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
            cv2.rectangle(frame, (x1, y1-25), (x1+len(cls)*12, y1), color, -1)
            cv2.putText(frame, cls+str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            print('{0} - {1}'.format(realtime, (cls, int(obj_id))))#, str(int(obj_id))))
                
            boxes.append([x1, y1, box_w, box_h])
            indexIDs.append(int(obj_id))
            memory[indexIDs[-1]] = boxes[-1]
            
        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    
                    p0 = (int(x + (w)/2), int(y + (h)/2))
                    p1 = (int(x2 + (w2)/2), int(y2 + (h2)/2))
                    cv2.line(frame, p0, p1, color, 3)
                    
                    if intersect(p0, p1, line_01[0], line_01[1]):
                        counter -=1
                        
                    elif intersect(p0, p1, line_02[0], line_02[1]):
                        counter +=1
                        
                i +=1
    
    # draw counter
    cv2.line(frame, line_01[0], line_01[1], (0,255,0), 4)
    cv2.line(frame, line_02[0], line_02[1], (0,255,0), 4)
    cv2.putText(frame, str(counter), (50,100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
    
    cv2.imshow('Stream', frame)
    #cv2.imwrite('output_video/frame-{}.png'.format(frames), frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(100)
    if ch == 27:
        break
    
outvideo.release()
cv2.destroyAllWindows()
