#!/usr/bin/env python3

import cv2
import numpy as np
from copy import deepcopy
from functions import Detection , Tracker ,recognition_model,recognition


def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    # Tracker Model
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create() 
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create() 
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create() 
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create() 
    # if tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

    # Recognition Model
    model = cv2.face.LBPHFaceRecognizer_create()

    # initialize variables
    tracker_counter = 0
    trackers = []
    iou_threshold = 0.8
    frame_counter=0

    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)

    # Tell general information for recognition functions
    path_to_training_images= '../data/at'
    training_image_size= (200, 200)
    names=[]
    unknown_count=0
    unknown_images=[]

    #Loops all the frames
    while True:
        # Read the frame
        ret, img = cap.read()
        #If the frame is invalid break the cycle
        if ret == False:
            break

        # Gets the timestamp
        stamp=cap.get(cv2.CAP_PROP_POS_MSEC)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_gui = deepcopy(img)
        detections=[]
        detection_counter = 0


        # Detect the faces
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 4)

        # Loops all the detected faces and draws a rectangle
        detections = []
        for bbox in faces: 
            x1, y1, w, h = bbox
            detection = Detection(x1, y1, w, h, gray, id=detection_counter)
            detection_counter += 1
            detections.append(detection)

        # ------------------------------------------
        # For each detection, see if there is a tracker to which it should be associated
        # ------------------------------------------
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                tracker_bbox = tracker.detections[-1]
                iou = detection.computeIOU(tracker_bbox)
                #print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection)
                    

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

        # ------------------------------------------
        # Create Tracker for each detection
        # ------------------------------------------
        if frame_counter == 0:
            for detection in detections:
                tracker = Tracker(detection, id=tracker_counter)
                tracker_counter += 1
                trackers.append(tracker)

        # ------------------------------------------
        # Draw stuff
        # ------------------------------------------

        # Draw trackers
        for tracker in trackers:
            tracker.draw(image_gui)

        # Draw all the detections
        for detection in detections:
            img = detection.draw(img)

        # Display the results
        cv2.imshow('img', img)
        # Stop if q key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()