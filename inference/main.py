#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:42:51 2019

@author: seungmin
"""


from area.area_designation import *
from readNet import *
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

# Draw area with the first frame of the video
vid = cv2.VideoCapture(opt.input_video)
mot_tracker = Sort()

cv2.namedWindow('Stream',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Stream', (800,600))

ret,frame=vid.read()
vw = frame.shape[1]
vh = frame.shape[0]

fourcc = cv2.VideoWriter_fourcc(*'MP4V') #fourcc = cv2.VideoWriter_fourcc(*'XVID')

outvideo = cv2.VideoWriter(opt.input_video.replace(".mp4", "-det.mp4"), fourcc, 20.0, (vw,vh), True)
print("\n")
print("Press 's' button to start design your area. Or 'p' button to pass this process.")


    # Initializing area_Designation class
CANVAS_SIZE = (int(vh), int(vw))
FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

area = PolygonDrawer("Area", frame, CANVAS_SIZE, FINAL_LINE_COLOR, WORKING_LINE_COLOR) 

#if cv2.waitKey(0) & 0xFF == ord("s"):
area.run()
    
#elif cv2.waitKey(0) & 0xFF == 27:
#    print("Starting object tracking without specific analysis area...")
#    pass
    
    
# Counting objects by intersection 
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
    
    detections = detect_image(my_model,
                              pilimg,
                              Tensor,
                              yolomodel["img_size"],
                              yolomodel["conf_thres"],
                              yolomodel["nms_thres"])
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (yolomodel["img_size"] / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (yolomodel["img_size"] / max(img.shape))
    unpad_h = yolomodel["img_size"] - pad_y
    unpad_w = yolomodel["img_size"] - pad_x
    
    boxes = []
    confidences = []
    classIDs = []
    
    # area designation
    overlay = frame.copy()
    pts = np.array(area.points, np.int32).reshape((-1,1,2))
    cv2.fillPoly(frame, [pts], (0,255,0))
    cv2.polylines(frame, [pts], False, (255,255,0), 3)
    opacity = 0.8
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    
    if detections is not None:
        
        tracked_objects = mot_tracker.update(detections.cpu())
        box = detections[0:4]
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
         
        boxes = []
        indexIDs = []
        memory = {}
        previous = memory.copy()        
        
        user_classes = ["person", "bicycle", "car", "bus", "motorbike", "truck"]
        
        
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])
            
            #p0 = (int(x1 + (box_w)/2), int(y1 + (box_h)/2))
            
            color = [int(c) for c in colors[int(cls_pred) % len(colors)]]
            cls = classes[int(cls_pred)]
            realtime = datetime.datetime.now().strftime("%H:%M:%S")
            
            # 사용자가 원하는 객체만 출력 
            if cls not in user_classes :
                pass
            else :
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 2)
                cv2.rectangle(frame, (x1, y1-25), (x1+len(cls)*12, y1), color, -1)
                #cv2.line(frame, p0, p0, color, 5)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            
            data_conf = np.array([realtime, cls, int(obj_id)]).reshape(3, -1)
            
            boxes.append([x1, y1, box_w, box_h])
            indexIDs.append(int(obj_id))
            memory[indexIDs[-1]] = boxes[-1]
        
        print(boxes)
        # 모든 박스에 대해서 중점좌표 계산
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
                    
                    cv2.line(frame, p0, p1, color, 4)
                    
                    #io = cv2.pointPolygonTest(pts, p1, False)
                    
                i +=1
    
    
    cv2.imshow('Stream', frame)
    outvideo.write(frame)
    ch = 0xFF & cv2.waitKey(100)
    if ch == 27:
        break

outvideo.release()
cv2.destroyAllWindows()
