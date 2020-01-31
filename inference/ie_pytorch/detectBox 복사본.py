#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:23:25 2020

@author: seungmin
"""

from tracker.sort import *
from ie_pytorch.readNet import *
from area.area_designation import *
from utils import *

import cv2
import datetime
from PIL import Image
import numpy as np


# Detecting and making tensor objects from each frame
def detect_image(model, img, tensor, img_size, conf_thres, nms_thres):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor()])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, conf_thres, nms_thres)
        
    return detections[0]


def unpad(img, img_size):
    img = np.array(img)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    return pad_x, pad_y, unpad_h, unpad_w, img


# Blending frames with designed area
def blending_area(frame, area_points):
    overlay = frame.copy()
    pts = np.array(area_points, np.int32).reshape((-1,1,2))
    cv2.fillPoly(frame, [pts], (0,255,0))
    cv2.polylines(frame, [pts], False, (255,255,0), 3)
    opacity = 0.8
    return cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def centroidBox(frame, boxes, indexes, previous, color):
    
    if len(boxes) > 0:
        i = int(0)
        for box in boxes:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
                
            if indexes[i] in previous:
                previous_box = previous[indexes[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                
                p0 = (int(x + (w)/2), int(y + (h)/2))
                p1 = (int(x2 + (w2)/2), int(y2 + (h2)/2))
                    
                cv2.line(frame, p0, p1, color, 4)
            
            i +=1
    #return p0, p1
                
        
def trackingBoxes(frame, detections, tracker, padx, pady, h, w, img, colors, classes):
    
    if detections is not None:
        objects = tracker.update(detections.cpu())
        
        boxes = []
        indexIDs = []
        #previous = memory.copy()
        memory = {}
        previous = memory.copy()
        
        user_classes = ["Person", "person", "bicycle", "Vehicle", "car", "Bus", "bus", "motorbike", "truck", "wheel",
                        "Motorcycle"]
        
        
        for x1, y1, x2, y2, obj_id, cls_pred in objects:
            
            box_h = int(((y2 - y1) / h) * img.shape[0])
            box_w = int(((x2 - x1) / w) * img.shape[1])
            y1 = int(((y1 - pady // 2) / h) * img.shape[0])
            x1 = int(((x1 - padx // 2) / w) * img.shape[1])
            
            p0 = (int(x1 + (box_w)/2), int(y1 + (box_h)/2))
            
            color = [int(c) for c in colors[int(cls_pred) % len(colors)]]
            cls = classes[int(cls_pred)]
            #realtime = datetime.datetime.now().strftime("%H:%M:%S")
            
            # 사용자가 원하는 객체만 출력 
            if cls not in user_classes :
                pass
            elif box_w > img.shape[1]/2 or box_h > img.shape[0]/2 :
                pass
            else :
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1+len(cls)*12, y1), color, -1)
                cv2.line(frame, p0, p0, color, 5)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            #data_conf = np.array([realtime, cls, int(obj_id)]).reshape(3, -1)
            
            boxes.append([x1, y1, box_w, box_h])
            indexIDs.append(int(obj_id))
            memory[indexIDs[-1]] = boxes[-1]
            
        #centroidBox(frame, boxes, indexIDs, previous, color)
            
    return boxes, indexIDs, previous, color


def readVid(input_vid, model, tensor, img_size, conf_thres, nms_thres, area, tracker, colors, classes):
    
    try:
        print("Loading video file...")
        vid = cv2.VideoCapture(input_vid)
    except:
        print("Please check your video file.")
        return
    
    
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    outvideo = cv2.VideoWriter(input_vid.replace(".mp4", "-det.mp4"),
                               fourcc,
                               fps,
                               (int(vid.get(3)),int(vid.get(4))),
                               True)
    
    print("Record start")
    
    frames = 0
    #frame_count = 0
    
    while(True):
        ret, frame = vid.read()
        if not ret:
            print("")
            break
        frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        
        detections = detect_image(model,
                                  pilimg,
                                  tensor,
                                  img_size,
                                  conf_thres,
                                  nms_thres)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        pad_x, pad_y, unpad_h, unpad_w, img_= unpad(pilimg,
                                                    img_size)
        
        blending_area(frame,
                      area.points)
        
        boxes, indexIDs, previous, color = trackingBoxes(frame,
                                                         detections,
                                                         tracker,
                                                         pad_x,
                                                         pad_y,
                                                         unpad_h,
                                                         unpad_w,
                                                         img_,
                                                         colors,
                                                         classes)
        
        cv2.imshow('Stream', frame)
        outvideo.write(frame)
        ch = 0xFF & cv2.waitKey(100)
        if ch == 27:
            break
        
    vid.release()
    outvideo.release()
    cv2.destroyAllWindows()