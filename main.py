#!/usr/bin/env python3

import cv2
import os
import numpy as np  
from detect_faces_functions import detect_faces_Haar
from face_recognition_functions import face_recognition


def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
            cv2.rectangle(detected_faces.img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detected_faces.extract_face(img,x, y, w, h)
            face_recognized=face_recognition()
            face_recognized.save_new_face(detected_faces.extracted_face,0)

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
