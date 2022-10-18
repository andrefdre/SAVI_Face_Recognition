import cv2
import csv
import numpy as np
from copy import deepcopy
from functions_2 import Detection , Tracker

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Count number of persons in the database
number_of_persons = 0
csv_reader = csv.reader(open('database_person.top'))
for row in csv_reader:
    if len(row) != 3: # skip badly formatted rows
        continue

    person_number, name , img_path = row
    
    person_number = int(person_number) # convert to number format (integer)
    if person_number >= number_of_persons:
        number_of_persons = person_number + 1

    # memorize the face
    #face = face_cascade.detectMultiScale(img_path,scaleFactor = 1.1, minNeighbors = 4)



# Create the colors for each person
colors = np.random.randint(0, high=255, size=(number_of_persons, 3), dtype=int)

# initialize variables
detection_counter = 0
tracker_counter = 0
trackers = []
iou_threshold = 0.8
frame_counter=0

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, img = cap.read()
    if ret == False:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_gui = deepcopy(img)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 4)

    # ------------------------------------------
    # Create Detections per haar cascade bbox
    # ------------------------------------------
    detections = []
    for bbox in faces: 
        x1, y1, w, h = bbox
        detection = Detection(x1, y1, w, h, gray, id=detection_counter)
        detection_counter += 1
        detection.draw(image_gui)
        detections.append(detection)
        #cv2.imshow('detection ' + str(detection.id), detection.image  )

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

    # Display
    cv2.imshow('img', image_gui)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()