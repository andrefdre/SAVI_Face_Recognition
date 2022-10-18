#!/usr/bin/env python3

import cv2
import os
import numpy as np  
from functions import detect_faces_Haar , detection ,face_recognition



def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = cv2.face.LBPHFaceRecognizer_create()
    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)

    trackings=[]

    while True:
        # Read the frame
        _, img = cap.read()

        #Initializes the detection class which gives all the detected faces in the frame
        bboxes=detect_faces_Haar(face_cascade, img, scaleFactor = 1.03,minNeighbors=7,minSize=(120, 120))

        detections=[]
        #Loops all the detected faces and draws a rectangle
        for bbox in bboxes.faces:
            (x, y, w, h) = bbox
            #Converts the frame to Gray Scale
            detected=detection(img,x,y,w,h)
            detections.append(detected)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        


            #Calls the recognition class that will try to recognize the face
            face_recognized=face_recognition()
            #face_recognized.save_new_face(detected.extracted_face,0)
            path_to_training_images = '../data/at'
            training_image_size = (200, 200)
            face_recognized.read_images(path_to_training_images, training_image_size)
            model.train(face_recognized.training_images, face_recognized.training_labels)
            roi_gray = gray[x:x+w, y:y+h]
            roi_gray = cv2.resize(roi_gray, training_image_size)
            label, confidence = model.predict(roi_gray)
            print(confidence,label)

        # Draw everything
        for detected in detections:
            img = detected.draw(img)
        
        # Display the results
        cv2.imshow('img', img)
        # Stop if q key is pressed
        if cv2.waitKey(1) == ord('q'):
                break

    #Release all windows
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
