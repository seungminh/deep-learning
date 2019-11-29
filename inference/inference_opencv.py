#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 10:09:16 2019

@author: seungmin
"""
import cv2
import numpy as np

import object_tracker as obt

# yolo path
yolomodel = {"config_path":"../trainer/PyTorch-YOLOv3/config/yolov3.cfg",
             "weights_path":"../trainer/PyTorch-YOLOv3/weights/yolov3.weights",
             "class_path":"../trainer/PyTorch-YOLOv3/data/coco.names",
             "conf_thres": 0.9,
             "nms_thres": 0.2
             }

# initialize network
net = cv2.dnn.readNetFromDarknet(yolomodel["config_path"],
                                 yolomodel["weights_path"])
labels = open(yolomodel["class_path"]).read().strip().split("\n")

np.random.seed(2020)
layer_names = net.getLayerNames()
layer_names = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
print(layer_names)

bbox_colors = np.random.randint(0, 255, size=(len(labels), 3))

maxLost=5
tracker = obt.Tracker(maxLost = maxLost)

video_path = "./input/pedestrian_06.mp4"
cap = cv2.VideoCapture(video_path)

# Starting object detection
(H, W) = (None, None)
writer = None

while(True):
    
    ret, frame = cap.read()
    
    if not ret:
        print("파일을 열 수 없습니다")
        break
    
    if W is None or H is None: (H, W) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections_layer = net.forward(layer_names)
    
    detections_bbox = []
    
    boxes, confidences, classIDs = [], [], []
    for out in detections_layer:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > yolomodel["conf_thres"]:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, yolomodel["conf_thres"], yolomodel["nms_thres"])
    
    if len(idxs)>0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            detections_bbox.append((x, y, x+w, y+h))
            clr = [int(c) for c in bbox_colors[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x+w, y+h), clr, 2)
            cv2.putText(frame, "{}:{:.4f}".format(labels[classIDs[i]], confidences[i]),
                        (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        
    # Update tracker
    objects = tracker.update(detections_bbox)
    
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
        
    cv2.imshow("image", frame)
    
    ch = 0xFF & cv2.waitKey(100)
    if ch == 27:
        break
    #if cv2.waitKey(1) & 0xFF ==ord('q'):
    #    break
    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output.mp4", fourcc, 30, (W, H), True)
    writer.write(frame)
    
writer.release()
cap.release()
cv2.destroyWindow("image")

    
    
        

