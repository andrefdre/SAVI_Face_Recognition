
#!/usr/bin/env python3

import csv
from copy import deepcopy

import cv2
import numpy as np


class BoundingBox:
    
    def __init__(self, x1, y1, w, h):
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.area = w * h

        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h


    def computeIOU(self, bbox2):
    
        x1_intr = min(self.x1, bbox2.x1)             
        y1_intr = min(self.y1, bbox2.y1)             
        x2_intr = max(self.x2, bbox2.x2)
        y2_intr = max(self.y2, bbox2.y2)

        w_intr = x2_intr - x1_intr
        h_intr = y2_intr - y1_intr
        A_intr = w_intr * h_intr

        A_union = self.area + bbox2.area - A_intr
        
        return A_intr / A_union



class Detection(BoundingBox):

    def __init__(self, x1, y1, w, h, image_full, id):
        super().__init__(x1,y1,w,h) # call the super class constructor        
        self.id = id
        self.extractSmallImage(image_full)

    def extractSmallImage(self, image_full):
        self.image = image_full[self.y1:self.y1+self.h, self.x1:self.x1+self.w]

    def draw(self, image_gui, color=(255,0,0)):
        cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)

        image = cv2.putText(image_gui, 'D' + str(self.id), (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)


class Tracker():

    def __init__(self, detection, id):
        self.detections = [detection]
        self.id = id


    def draw(self, image_gui, color=(255,0,255)):
        last_detection = self.detections[-1] # get the last detection

        cv2.rectangle(image_gui,(last_detection.x1,last_detection.y1),
                      (last_detection.x2, last_detection.y2),color,3)

        image = cv2.putText(image_gui, 'T' + str(self.id), 
                            (last_detection.x2-40, last_detection.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, color, 2, cv2.LINE_AA)

    def addDetection(self, detection):
        self.detections.append(detection)


    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return 