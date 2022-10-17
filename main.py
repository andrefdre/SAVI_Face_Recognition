#!/usr/bin/env python3

import cv2
import os
import numpy as np  
from detect_faces_functions import detect_faces_Haar
from face_recognition_functions import face_recognition


def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = cv2.face.LBPHFaceRecognizer_create()
    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        _, img = cap.read()

        #Calls the function that will detect the faces
        #Initializes the detection class
        detected_faces=detect_faces_Haar(face_cascade, img, scaleFactor = 1.03,minNeighbors=7,minSize=(120, 120))
        #Loops all the detected faces and draws a rectangle
        for (x, y, w, h) in detected_faces.faces:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi_gray = gray[x:x+w, y:y+h]
            if roi_gray.size == 0:
                # The ROI is empty. Maybe the face is at the image edge.
                # Skip it.
                continue
            cv2.rectangle(detected_faces.img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detected_faces.extract_face(img,x, y, w, h)
            face_recognized=face_recognition()
            face_recognized.save_new_face(detected_faces.extracted_face,0)
            path_to_training_images = '../data/at'
            training_image_size = (200, 200)
            face_recognized.read_images(path_to_training_images, training_image_size)
            model.train(face_recognized.training_images, face_recognized.training_labels)
            roi_gray = gray[x:x+w, y:y+h]
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)

        # Display the results
        cv2.imshow('img', detected_faces.img_copy)
        # Stop if q key is pressed
        if cv2.waitKey(1) == ord('q'):
                break

    #Release all windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
