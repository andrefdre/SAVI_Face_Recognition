#!/usr/bin/env python3

import cv2
import numpy as np
import os

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
    def __init__(self, full_image , x, y , w , h, id):
        super().__init__(x, y, w, h)
        self.img_gui=full_image
        self.extracted_face=self.extract_face(full_image , x, y , w , h)
        self.id=id

    def extract_face(self,image,x, y, w, h):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        cv2.imshow("Extracted Face",gray[y:y+h, x:x+w])
        return gray[y:y+h, x:x+w]
        
    def draw(self,image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.rectangle(image, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
        image = cv2.putText(image,'Det ' + str(self.id),(self.x1,self.y1-15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return image



class recognition_model():
    def __init__(self,path,image_size): 
        self.names,self.training_images,self.training_labels=self.read_images(path,image_size)

    def save_new_face(self,cutted_faces,name):
        output_folder = '../data/at/' + str(name)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        count=0
        print(len(cutted_faces))
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

    def draw(self,image,name):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,'Name: ' + str(name),(self.detection.x1,self.detection.y1-35), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    