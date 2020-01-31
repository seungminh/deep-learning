#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:05:19 2020

@author: seungmin
"""

import cv2
import numpy as np


def pointAreaTest(area_points, box_points):
    
    #inside = []
    for point in box_points:
        dist = cv2.pointPolygonTest(np.array(area_points, np.float32).reshape((-1,1,2)),
                                    point,
                                    measureDist=False)
        print(dist)
        
    #return dist