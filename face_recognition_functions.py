#!/usr/bin/env python3

import cv2
import numpy as np
import os

class face_recognition():
    def __init__(self): 
        print("not yet done")

    def save_new_face(self,cutted_face,count=0):
        output_folder = '../data/at/jm'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        gray = cv2.cvtColor(cutted_face, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(gray, (200, 200))
        face_filename = '%s/%d.pgm' % (output_folder, count)
        cv2.imwrite(face_filename, face_img)

    def read_images(self,path, image_size):
        self.names = []
        self.training_images, self.training_labels = [], []
        label = 0
        for dirname, subdirnames, filenames in os.walk(path):
            for subdirname in subdirnames:
                self.names.append(subdirname)
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    img = cv2.imread(os.path.join(subject_path, filename),cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        # The file cannot be loaded as an image.
                        # Skip it.
                        continue
                    img = cv2.resize(img, image_size)
                    self.training_images.append(img)
                    self.training_labels.append(label)
                    label += 1
        self.training_images = np.asarray(self.training_images, np.uint8)
        self.training_labels = np.asarray(self.training_labels, np.int32)

