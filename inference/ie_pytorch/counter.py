#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:05:19 2020

@author: seungmin
"""

import cv2
import numpy as np


def pointAreaTest(area_points, box_points):
    
    inside = []
    for point in box_points:
        dist = cv2.pointPolygonTest(np.array(area_points, np.float32).reshape((-1,1,2)),
                                    point,
                                    measureDist=False)
        if int(dist) == 1:
            inside.append(dist)
        
    return int(len(inside))

# Counting objects by intersection 
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])