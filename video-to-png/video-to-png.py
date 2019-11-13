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
from sys import argv
import shutil

cap = cv2.VideoCapture(argv[1],0)
frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(frames)

counter=0
counter2=0

a,b = str(argv[1]).split('.')
path = a

if len(argv)>2:
	fe=int(argv[2])
else:
	fe=1
if os.path.exists(a):
	shutil.rmtree(path)

os.mkdir(path)
f = open("./"+path+"/"+"Readme.txt", "w+")
f.write("1. Move all images & name.txt file to your custom dataset folder." + '\n')
f.write("2. Split train, valid dataset with your name.txt file.")
f.close()

#f = open("."+argv[3]+"/"+"name.txt", "w+")
while(True):
	ret, frame = cap.read()
	counter +=1
	if not ret:break
	if (counter%fe!=0):
		
		continue
	else:
		counter2+=1
		cv2.imwrite("./"+path+"/"+str(counter2)+".jpg",frame)
		print("totoal: "+str(round(counter/frames*100,3))+"% || "+ "output image "+str(counter2)+ " / "+str(round(frames/fe,0)))

	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
# When everything done, release the capture
f.close()
cap.release()
cv2.destroyAllWindows()
