#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:35:35 2020

@author: seungmin
"""


import numpy as np
import cv2


class PolygonDrawer(object):
    
    def __init__(self, window_name, frame, CANVAS_SIZE, FINAL_LINE_COLOR, WORKING_LINE_COLOR):
        self.window_name = window_name 
        self.frame = frame
        self.CANVAS_SIZE = CANVAS_SIZE
        self.FINAL_LINE_COLOR = FINAL_LINE_COLOR
        self.WORKING_LINE_COLOR = WORKING_LINE_COLOR
        self.done = False
        self.done_all = False
        self.current = (0, 0)
        self.points = []
        self.polygons = []

    def on_mouse(self, event, x, y, buttons, user_param):
        
        if event == cv2.EVENT_MOUSEMOVE:
            self.current = (x, y)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            print("Completing polygon with %d points." % len(self.points))
            self.done = True
            
        if self.done:
            return
            

    def run(self):
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, self.frame)
        cv2.waitKey(1)
        canvas = self.frame
        
        while(not self.done_all):
            
            canvas = canvas.copy()
            cv2.setMouseCallback(self.window_name, self.on_mouse)
            
            if (len(self.points) > 0):
                cv2.polylines(canvas, np.array([self.points]), False, self.FINAL_LINE_COLOR, 2)
                cv2.line(canvas, self.points[-1], self.current, self.WORKING_LINE_COLOR)
                
            cv2.imshow(self.window_name, canvas)
            
            if (len(self.points) > 0):
                cv2.fillPoly(canvas, np.array([self.points]), self.FINAL_LINE_COLOR)
            
            if cv2.waitKey(1) & 0xFF == 27:
                self.done_all = True
            
        cv2.destroyWindow(self.window_name)
        return canvas
