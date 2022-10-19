#!/usr/bin/env python3

# Import all the libraries required
import cv2
import numpy as np
from copy import deepcopy
from functions import Detection , Tracker ,recognition_model,recognition


def main():
    # Load the cascade model for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

    # Recognition Model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Initialize variables
    tracker_counter = 0
    trackers = []
    detection_counter = 0
    # Threshold for the relation of the detection and tracker
    iou_threshold = 0.8
    # Tell general information for recognition functions
    path_to_training_images= '../data/at'
    training_image_size= (200, 200)
    names=[]
    unknown_count=0
    unknown_images=[]

    # Capture the video from webcam
    cap = cv2.VideoCapture(0)

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
        # Create a copy of the image so we can do alterations to it and still preserve the original image
        image_gui = deepcopy(img)
        # Create a list of detections and a counter that resets every cycle
        detections=[]

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 4)

        # Loops all the detected faces and Creates a detection and adds it to detection array
        for bbox in faces: 
            x1, y1, w, h = bbox
            # Initializes the Detector
            detection = Detection(x1, y1, w, h, gray, id=detection_counter)
            detection_counter += 1
            detections.append(detection)

        # Loops all the detections and loops all the trackers and computes if they overlap and if they do add the new detection to the tracker
        for detection in detections: # cycle all detections
            for tracker in trackers: # cycle all trackers
                # Gets the last detection in the tracker to compute its overlap
                tracker_bbox = tracker.detections[-1]
                # Computes the overlap of both bboxes
                iou = detection.computeIOU(tracker_bbox)
                # If both bboxes overlap add the detection to the tracker
                if iou > iou_threshold: # associate detection with tracker 
                    tracker.addDetection(detection, gray)

        # Loops all the trackers and checks if any of the new detections is associated to the tracker if not update tracker  
        for tracker in trackers: # cycle all trackers
            # Gets the last detection ID in the tracker
            last_detection_id = tracker.detections[-1].id
            # Gets all the IDs of the Detections
            detection_ids = [d.id for d in detections]
            # If the last id in the tracker is not one of the new Detection update Tracker
            if not last_detection_id in detection_ids:
                # Update Tracker
                tracker.updateTracker(gray)


        # Creates new trackers if the Detection has no tracker associated
        for detection in detections:
            # Checks to see if the Detections have a tracker associated to them
            if not detection.assigned_to_tracker:
                # Initializes the tracker
                tracker = Tracker(detection, id=tracker_counter, image=gray)
                tracker_counter += 1
                trackers.append(tracker)
                    

        #Constructs the class for recognition model
        recognitionModel =recognition_model(path_to_training_images, training_image_size)
        #Trains the model
        model.train(recognitionModel.training_images, recognitionModel.training_labels)

        #Loops all the detections and finds the person
        for detect in detections:
            roi_gray = cv2.resize(detect.extracted_face, training_image_size)
            label, confidence = model.predict(roi_gray)
            recon=recognition(detect)
            if confidence<70:
                image_gui = recon.draw(image_gui,recognitionModel.names[label],confidence)
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
        # Draw stuff
        # ------------------------------------------

        # Draw trackers 
        for tracker in trackers:
            image_gui = tracker.draw(image_gui)

        # Draw all the detections
        for detection in detections:
            img = detection.draw(image_gui)

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