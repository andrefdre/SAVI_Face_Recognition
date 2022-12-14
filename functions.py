#!/usr/bin/env python3

from gtts import gTTS
from playsound import playsound
import os
import cv2
import numpy as np
from turtle import color

###################################
# Bounding Box Class              #
###################################
class BoundingBox:
    # Function that initializes the Bounding Boxes
    def __init__(self, x1, y1, w, h):
        # Stores all the local variables
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        # Computes the are of the bbox to use in the IOU method
        self.area = w * h
        # Calculates the other corner coordinates
        self.x2 = self.x1 + self.w
        self.y2 = self.y1 + self.h

    # Function that will compute the intersection of both bboxes
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

    # Function that will extract the image inside the bounding box
    def extractSmallImage(self, image_full):
        return image_full[int(self.y1):int(self.y1+self.h), int(self.x1):int(self.x1+self.w)]


###########################################
# Detector Class                          #
###########################################
class Detection(BoundingBox):
    # Function that will initialize the Detector
    def __init__(self, x1, y1, w, h , image_full, stamp,id):
        # Calls the super class constructor
        super().__init__(x1,y1,w,h)
        # Stores the id 
        self.id = id
        # Extracts the image inside the detected image
        self.extracted_image = self.extractSmallImage(image_full)
        # Initializes the variable that will tell if has a tracker associated
        self.assigned_to_tracker=False
        self.stamp=stamp

    # Function that will draw the detection
    def draw(self, image_gui, color=(255,0,0)):
        # Draws the rectangle around the detected part of the image
        image_gui = cv2.rectangle(image_gui,(self.x1,self.y1),(self.x2, self.y2),color,3)
        # Writes some information about the detection
        image_gui = cv2.putText(image_gui, 'D' + str(self.id), (self.x1, self.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # Returns the image to be shown
        return image_gui


###########################################
# Detector Class                          #
###########################################
class Detection_face(BoundingBox):
    # Function that will initialize the Detector
    def __init__(self, x1, y1, w, h , bbox_tracker , image_full, id ):
        # Calls the super class constructor
        super().__init__(x1,y1,w,h)
        # Stores the id 
        self.id = id
        # Extracts the image inside the detected image
        self.extracted_image = self.extractSmallImage(image_full)
        # Stores the Bounding Box of the tracker so it draws in the correct position
        self.bbox_tracker=bbox_tracker

    # Function that will draw the detection
    def draw(self, image_gui, color=(255,0,0)):
        # Draws the rectangle around the detected part of the image
        image_gui = cv2.rectangle(image_gui,(self.x1+self.bbox_tracker.x1,self.y1+self.bbox_tracker.y1),(self.x2+self.bbox_tracker.x1, self.y2+self.bbox_tracker.y1),color,3)
        # Writes some information about the detection
        image_gui = cv2.putText(image_gui, 'D' + str(self.id), (self.x1+self.bbox_tracker.x1, self.y1+self.bbox_tracker.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        # Returns the image to be shown
        return image_gui

############################################
# Tracker Class                            #
############################################
class Tracker():
    # Function that initializes the Tracker
    def __init__(self, detection, id,image):
        # Creates an array to keep tracking of all the detections
        self.detections = []
        # Creates an array of Bounding boxes to later draw them
        self.bboxes = []
        # Initializes the tracker model
        self.tracker = cv2.TrackerCSRT_create()
        # Template image of the detected face
        self.template = None
        # Face inside the tracker
        self.face= None
        # Gives an ID to the tracker
        self.id = id
        # Name of the person
        self.name=None
        # Status of the tracker
        self.active=True
        # Initializes the tracker and associates the detection
        self.addDetection(detection,image)


    # Function that will Draw the tracker based on the last bbox
    def draw(self, image_gui, color=(255,0,255)):
        # Gets the last Bounding Box to use its coordinates 
        bbox = self.bboxes[-1] # get last bbox
        # If the tracker never disapeared from frame draw it
        if self.active==True:
            # Draws the rectangle
            image_gui = cv2.rectangle(image_gui,(bbox.x1,bbox.y1),(bbox.x2, bbox.y2),color,3)
            # Puts the Tracking ID
            image_gui = cv2.putText(image_gui, 'T' + str(self.id), (bbox.x2-40, bbox.y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            # Writes the person name
            image_gui = cv2.putText(image_gui,'Name: ' + str(self.name),(bbox.x1,bbox.y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            # Returns the modified image
        return image_gui

    # Function that will add a Detection to the tracker so it can be later used to update itself
    def addDetection(self, detection,image):
        #Initializes the tracker
        self.tracker.init(image, (detection.x1, detection.y1, detection.w, detection.h))
        #Adds the last detection to the tracker
        self.detections.append(detection)
        self.template = detection.extracted_image
        #Sets the detection to have a tracker assigned
        detection.assigned_to_tracker = True
        bbox = BoundingBox(detection.x1, detection.y1, detection.w, detection.h)
        self.bboxes.append(bbox)

    # Function that will update the tracker if no detection is associated to the tracker
    def updateTracker(self,image_gray):
        # Calls the tracker model to update the tracer
        ret, bbox = self.tracker.update(image_gray)
        # Creates a new Bounding Box since the bbox given by the tracker as a different construction than what we use
        x1,y1,w,h = bbox
        bbox = BoundingBox(int(x1), int(y1), int(w), int(h))
        # Appends the bbox to be used in the Drawing
        self.bboxes.append(bbox)
        # Update template using new bbox coordinates
        self.template = bbox.extractSmallImage(image_gray)

    def __str__(self):
        text =  'T' + str(self.id) + ' Detections = ['
        for detection in self.detections:
            text += str(detection.id) + ', '
        return text

    def updateTime(self, stamp):
        self.time_since_last_detection = round(stamp-self.getLastDetectionStamp(),1)

        if self.time_since_last_detection > 2: # deactivate tracker        
            self.active = False

    def getLastDetectionStamp(self):
        return self.detections[-1].stamp

#########################################
# Recognition Class                     #
#########################################
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
    def __init__(self,tracker):
        self.tracker=tracker

    def draw(self,image,name,confidence):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bbox = self.tracker.bboxes[-1] # get last bbox
        image = cv2.putText(image,'Name: ' + str(name),(bbox.x1,bbox.y1-35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        image = cv2.putText(image,'Confidence: ' + str(confidence),(bbox.x1,bbox.y1-55), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return image

##################################
# Speak Class                    #
##################################

class Speak():
    def __init__(self,text):
        self.text = text
        tts = gTTS(text, lang = 'pt')
        tts.save("./hi.mp3")
        playsound('./hi.mp3')
    
    

