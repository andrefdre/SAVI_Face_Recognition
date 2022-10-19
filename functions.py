#!/usr/bin/env python3

import os
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

    def extractSmallImage(self, image_full):
        self.extracted_face = image_full[self.y1:self.y1+self.h, self.x1:self.x1+self.w]



class Detection(BoundingBox):

    def __init__(self, x1, y1, w, h, image_full, id):
        super().__init__(x1,y1,w,h) # call the super class constructor        
        self.id = id
        self.extractSmallImage(image_full)
        self.assigned_to_tracker=False

    def draw(self, image_gui, color=(255,0,0)):
        cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)

        image = cv2.putText(image_gui, 'D' + str(self.id), (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        return image

class Tracker():

    def __init__(self, detection, id,image):
        # Creates an array to keep tracking of all the detections
        self.detections = [detection]
        # Creates an array of Bounding boxes to later draw them
        self.bboxes = []
        # Initializes the tracker model
        self.tracker = cv2.TrackerCSRT_create()
        # Gives an ID to the tracker
        self.id = id
        # Initializes the tracker and associates the detection
        self.addDetection(detection,image)


    def draw(self, image_gui, color=(255,0,255)):
        last_detection = self.detections[-1] # get the last detection

        bbox = self.bboxes[-1] # get last bbox

        print(bbox.x1,bbox.y1,bbox.x2,bbox.y2)

        image_gui = cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),color,3)

        image_gui = cv2.putText(image_gui, 'T' + str(self.id), (bbox.x2-40, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        return image_gui

    def addDetection(self, detection,image):
        #Initializes the tracker
        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))
        #Adds the last detection to the tracker
        self.detections.append(detection)
        #Sets the detection to have a tracker assigned
        detection.assigned_to_tracker = True
        bbox = BoundingBox(detection.x1, detection.y1, detection.w, detection.h)
        self.bboxes.append(bbox)

    def updateTracker(self,image_gray):
         ret, bbox = self.tracker.update(image_gray)
         # Creates a new Boundix Box since the bbox given by the tracker as a different construction than what we use
         x1,y1,w,h = bbox
         bbox = BoundingBox(int(x1), int(y1), int(w), int(h))
         self.bboxes.append(bbox)

    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '

        return text


class recognition_model():
    def __init__(self,path,image_size): 
        self.names,self.training_images,self.training_labels=self.read_images(path,image_size)

    def save_new_face(self,cutted_faces,name):
        output_folder = '../data/at/' + str(name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        count=0
        for cutted_face in cutted_faces:
            face_img = cv2.resize(cutted_face, (200, 200))
            face_filename = '%s/%d.pgm' % (output_folder, count)
            cv2.imwrite(face_filename, face_img)
            count+=1

    def read_images(self,path, image_size):
        names = []
        training_images, training_labels = [], []
        label = 0
        for dirname, subdirnames, filenames in os.walk(path):
            for subdirname in subdirnames:
                names.append(subdirname)
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    img = cv2.imread(os.path.join(subject_path, filename),cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        # The file cannot be loaded as an image.
                        # Skip it.
                        continue
                    img = cv2.resize(img, image_size)
                    training_images.append(img)
                    training_labels.append(label)
                label += 1
        training_images = np.asarray(training_images, np.uint8)
        training_labels = np.asarray(training_labels, np.int32)
        return names, training_images, training_labels

class recognition():
    def __init__(self,detection):
        self.detection=detection

    def draw(self,image,name,confidence):
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(image,'Name: ' + str(name),(self.detection.x1,self.detection.y1-35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image,'Confidence: ' + str(confidence),(self.detection.x1,self.detection.y1-55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return image
    

