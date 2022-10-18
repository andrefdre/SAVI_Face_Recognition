#!/usr/bin/env python3

import cv2

class bounding_box():
    def __init__(self,x,y,w,h):
        self.x1=x
        self.y1=y
        self.w=w
        self.h=h
        self.x2=self.x1+self.w
        self.y2=self.y1+self.h

class detect_faces_Haar():
    def __init__(self,f_cascade, colored_img, scaleFactor = 1.2,minNeighbors=5,minSize=(50, 50)):
        #just making a copy of image passed, so that passed image is not changed
        self.img_copy = colored_img.copy()
        #convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(self.img_copy, cv2.COLOR_BGR2GRAY)       
        #let's detect multiscale (some images may be closer to camera than others) images
        self.faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors,minSize=minSize);   
        #go over list of faces and draw them as rectangles on original colored img

    
class detection(bounding_box):
    def __init__(self, full_image , x, y , w , h):
        super().__init__(x, y, w, h)
        self.extracted_face=self.extract_face(full_image , x, y , w , h)

    def extract_face(self,image,x, y, w, h):
        cv2.imshow("Extracted Face",image[y:y+h, x:x+w])
        return image[y:y+h, x:x+w]
        
    def draw(self):
        cv2.rectangle(detected_faces.img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img,'OpenCV',(10,500), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
