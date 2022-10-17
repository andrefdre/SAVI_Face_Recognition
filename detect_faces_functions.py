#!/usr/bin/env python3

import cv2

class detect_faces_Haar():
    def __init__(self,f_cascade, colored_img, scaleFactor = 1.2,minNeighbors=5,minSize=(50, 50)):
        #just making a copy of image passed, so that passed image is not changed
        self.img_copy = colored_img.copy()
        #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)       
        #let's detect multiscale (some images may be closer to camera than others) images
        self.faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,minSize=minSize);   
        #go over list of faces and draw them as rectangles on original colored img


    def extract_face(self,image,x, y, w, h):
        self.extracted_face=image[y:y+h, x:x+w]
        cv2.imshow("Extracted Face",self.extracted_face)
        
