#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:13:24 2019

@author: seungmin
"""


import numpy as np
import cv2
import math
import os
import sys
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, help="Input video file name with its path")
    parser.add_argument("--fps", type=int, default=25, help="fps")
    parser.add_argument("--custom_data_path", type=str, default="/please/change/this/path", help="Path to the trainer of YOLO")
    opt = parser.parse_args()
    print(opt)

#

cap = cv2.VideoCapture(opt.input_video,0)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("이 비디오는 총 {0}장의 프레임으로 구성되어 있습니다.".format(int(frames)))
print("{0} 프레임마다 비디오를 나누어 캡쳐합니다.".format(opt.fps))

counter=0
counter2=0

a,b = str(opt.input_video).split('.')
path = a

fe=opt.fps

if os.path.exists(a):
	shutil.rmtree(path)

os.mkdir(path)


while(True):
	ret, frame = cap.read()
	counter +=1
	if not ret:break
	if (counter%fe!=0):
		continue
	else:
		counter2+=1
		cv2.imwrite("./"+ path +"/"+ path.split('/')[-1] +"_"+str(counter2)+".jpg",frame)
		print("totoal: "+str(round(counter/frames*100,3))+"% || "+ "output image "+str(counter2)+ " / "+str(round(frames/fe,0)))

	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()


image_list = os.listdir("./"+path)
image_list = sorted(image_list, key=lambda x: int(x.partition('_')[-1].partition('.')[0]))

custom_data_path = opt.custom_data_path

f = open("./"+path+"/"+"name.txt", "w+")

for image_name in image_list:
    f.write(custom_data_path+"/"+image_name + '\n')
f.close()

f = open("./"+path+"/"+"Readme.txt", "w+")
f.write("1. Move all images & name.txt file to your custom dataset folder." + '\n')
f.write("2. Split train, valid dataset with your name.txt file.")
f.close()
