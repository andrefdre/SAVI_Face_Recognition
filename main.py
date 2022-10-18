#!/usr/bin/env python3

import cv2
import os
import numpy as np  
from functions import detect_faces_Haar , detection ,recognition_model,recognition



def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = cv2.face.LBPHFaceRecognizer_create()
    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)

    # Tell general information for recognition functions
    path_to_training_images= '../data/at'
    training_image_size= (200, 200)
    names=[]
    unknown_count=0
    unknown_images=[]

    trackings=[]

    while True:
        # Read the frame
        _, img = cap.read()
        print(cap.get(cv2.CAP_PROP_POS_MSEC))

        #Initializes the detection class which gives all the detected faces in the frame
        bboxes=detect_faces_Haar(face_cascade, img, scaleFactor = 1.03,minNeighbors=7,minSize=(120, 120))

        detections=[]
        detections_id=0
        #Loops all the detected faces and draws a rectangle
        for bbox in bboxes.faces:
            (x, y, w, h) = bbox
            #Converts the frame to Gray Scale
            detected=detection(img,x,y,w,h,detections_id)
            detections.append(detected)
            detections_id+=1
        


        ############################################
        # Recognition
        ###########################################

        #Constructs the class for recognition 
        recognitionModel =recognition_model(path_to_training_images, training_image_size)
        #Trains the model
        model.train(recognitionModel.training_images, recognitionModel.training_labels)

        #Loops all the detections and finds the person
        for detect in detections:
            roi_gray = cv2.resize(detect.extracted_face, training_image_size)
            label, confidence = model.predict(roi_gray)
            recon=recognition(detect)
            if confidence<70:
                recon.draw(img,recognitionModel.names[label],confidence)
            else:
                recon.draw(img,'Unknown','NAN')
                unknown_count+=1
                unknown_images.append(detect.extracted_face)
                if unknown_count>10:
                    print("What's the person's name: ")
                    name=input()
                    recognitionModel.save_new_face(unknown_images,name)
                cv2.imshow('unknown', detect.extracted_face)

        # Draw all the detections
        for detect in detections:
            img = detect.draw(img)
        
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
