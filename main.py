#!/usr/bin/env python3

# Import all the libraries required
import mediapipe as mp
import threading
import pyttsx3
import cv2
import numpy as np
from copy import deepcopy
from functions import Detection, Detection_face , Tracker ,recognition_model,recognition, Speak
from detector import Detector
#from lib import VisTrack, show_video, create_video

def main():
    # Load the body detector
    model_path = 'model/model.tflite'
    input_shape = [192,192]
    score_th = 0.4
    nms_th = 0.5
    num_threads = None

    # Initializes the model detector class
    detector = Detector(
    model_path=model_path,
    input_shape=input_shape,
    score_th=score_th,
    nms_th=nms_th,
    providers=['CPUExecutionProvider'],
    num_threads=num_threads,
    )

    # Load the cascade model for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_alt.xml')

    # Recognition Model
    model = cv2.face.LBPHFaceRecognizer_create()

    # Initialize variables
    tracker_counter = 0
    trackers = []
    detection_counter = 0
    face_detection_counter=0
    # Threshold for the relation of the detection and tracker
    iou_threshold = 0.9
    border=30
    disabling_offset=20
    # Tell general information for recognition functions
    path_to_training_images= '../data/at'
    training_image_size= (200, 200)
    unknown_count=0
    unknown_images=[]
    stamp_since_last_unknown_image=0
    #Constructs the class for recognition model
    recognitionModel =recognition_model(path_to_training_images, training_image_size)
    #Trains the model
    model.train(recognitionModel.training_images, recognitionModel.training_labels)

    # Capture the video from webcam
    cap = cv2.VideoCapture(0)

    #Text to speech
    engine = pyttsx3.init()

    # Body Recognition
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        #Loops all the frames
        while cap.isOpened():
            # Read the frame
            ret, img = cap.read()
            # Gets Image size
            height,width,_ = img.shape
            #If the frame is invalid break the cycle
            if ret == False:
                break

            # Gets the timestamp
            stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Create a copy of the image so we can do alterations to it and still preserve the original image
            image_gui = deepcopy(img)

            # Creates a border in the image to simbolize the limits of the trackers
            image_darkned=(image_gui*0.5).astype(np.uint8)
            image_darkned[border:height-border,border:width-border]=image_gui[border:height-border,border:width-border]
            image_gui=image_darkned

            # Detect the bodies
            bboxes, scores, class_ids = detector.inference(img)
            # Create a list of detections and a counter that resets every cycle
            detections=[]
            # Loops all the detected bodies and Creates a detection and adds it to detection array
            for idx , bbox in enumerate(bboxes):
                if(scores[idx]>0.8): 
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    w=x2-x1
                    h=y2-y1
                    # Initializes the Detector
                    if not w*h< 20000:
                        # Initializes the class detection
                        detection = Detection(x1, y1, w, h, gray, id=detection_counter)
                        # Increases the detection counter
                        detection_counter += 1
                        # Adds the detection to the array
                        detections.append(detection)

            # Loops all the detections and loops all the trackers and computes if they overlap and if they do add the new detection to the tracker
            for detection in detections: # cycle all detections
                for tracker in trackers: # cycle all trackers
                    # Gets the last detection in the tracker to compute its overlap
                    tracker_bbox = tracker.bboxes[-1]
                    # Computes the overlap of both bboxes
                    iou = detection.computeIOU(tracker_bbox)
                    # If both bboxes overlap add the detection to the tracker
                    print('IOU( T' + str(tracker.id) + ' D' + str(detection.id) + ' ) = ' + str(iou))
                    if iou > iou_threshold: 
                        # Associate detection with tracker 
                        tracker.addDetection(detection, gray)

            ##############################################
            # Does all the processing to each tracker    #
            ##############################################

            for tracker in trackers: # cycle all trackers
                # Will check if the tracker overlaps with the detection and if so add detection otherwise do tracking
                # Checks if the tracker is active so it doesn't use unnecessary processor
                if tracker.active==True:
                    # Gets the last detection ID in the tracker
                    last_detection_id = tracker.detections[-1].id
                    # Gets all the IDs of the Detections
                    detection_ids = [d.id for d in detections]
                    # If the last id in the tracker is not one of the new Detection update Tracker
                    if not last_detection_id in detection_ids:
                        # Update Tracker
                        tracker.updateTracker(gray)

            # ###################################
            # # Body Recognition                #
            # ###################################
            # # Loops all the trackers and if they are active do body feature recognition 
            # for tracker in trackers: # cycle all trackers
            #     if tracker.active==True:
            #         bbox = tracker.bboxes[-1] # get last bbox
            #         # If the tracker center is outside the border set's it to deactivated
            #         # if bbox.x1+disabling_offset<border or bbox.x2-disabling_offset>width-border or bbox.y1+disabling_offset <border:
            #         #     tracker.active=False
            #         tracker_image=image_gui[int(bbox.y1):int(bbox.y2),int(bbox.x1):int(bbox.x2),:]
            #         if tracker_image.shape[0]>0 and tracker_image.shape[1]>0:
            #             # Converts to RGB for the holistic process
            #             tracker_image = cv2.cvtColor(tracker_image, cv2.COLOR_BGR2RGB)
            #             # Make Detections of the body features
            #             results = holistic.process(tracker_image)          
            #             # Recolor image back to BGR for rendering
            #             tracker_image = cv2.cvtColor(tracker_image, cv2.COLOR_RGB2BGR)

            #             mp_drawing.draw_landmarks(tracker_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
            #             # Right hand
            #             mp_drawing.draw_landmarks(tracker_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            #                 mp_drawing.DrawingSpec(color=(170,44,121), thickness=2, circle_radius=2))

            #             # Left Hand
            #             mp_drawing.draw_landmarks(tracker_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            #                 mp_drawing.DrawingSpec(color=(78,44,214), thickness=2, circle_radius=4),
            #                 mp_drawing.DrawingSpec(color=(124,67,32), thickness=2, circle_radius=2))

            #             # Pose Detections
            #             mp_drawing.draw_landmarks(tracker_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            #                 mp_drawing.DrawingSpec(color=(78,44,214), thickness=2, circle_radius=4),
            #                 mp_drawing.DrawingSpec(color=(124,67,32), thickness=2, circle_radius=2))
            #             image_gui[int(bbox.y1):int(bbox.y2),int(bbox.x1):int(bbox.x2),:]=tracker_image

            # #######################################
            # # Face Detection                      #
            # #######################################
            # face_detections=[]
            # for tracker in trackers: # cycle all trackers
            #     if tracker.active==True:
            #         # Detect the faces
            #         faces = face_cascade.detectMultiScale(tracker.template,scaleFactor = 1.1, minNeighbors = 4)
            #         # Loops all the detected faces and Creates a detection and adds it to detection array
            #         for bbox in faces: 
            #             x1, y1, w, h = bbox
            #             bbox_tracker=tracker.bboxes[-1]
            #             # Initializes the Detector
            #             if not w*h< 5000:
            #                 face_detection = Detection_face(x1, y1, w, h ,bbox_tracker , tracker.template, id=face_detection_counter)
            #                 face_detection_counter += 1
            #                 face_detections.append(face_detection)
            #                 tracker.face=face_detection.extracted_image


            # Creates new trackers if the Detection has no tracker associated
            for detection in detections:
                # Checks to see if detection is inside allowed frame so it doesn't create trackers that will be deleted
                #if not detection.x1+disabling_offset<border or detection.x2-disabling_offset>width-border or detection.y1 + disabling_offset <border:
                    # Checks to see if the Detections have a tracker associated to them
                if not detection.assigned_to_tracker:
                    # Initializes the tracker
                    tracker = Tracker(detection, tracker_counter, gray)
                    tracker_counter += 1
                    trackers.append(tracker)
                    

            # ######################################
            # # Face recognition                   #
            # ######################################
            # for tracker in trackers:
            #     if tracker.name==None and tracker.active:
            #         if not tracker.face is None:
            #             roi_gray = cv2.resize(tracker.face, training_image_size)
            #             label, confidence = model.predict(roi_gray)
            #             if confidence<60:
            #                 text = f"Hello " + str(recognitionModel.names[label])
            #                 threading.Thread(target=Speak, args=(text,)).start()
            #                 tracker.name=recognitionModel.names[label]
            #                 recon=recognition(tracker)
            #                 image_gui = recon.draw(image_gui,recognitionModel.names[label],confidence)
            #             else:
            #                 unknown_count+=1
            #                 unknown_images.append(tracker.face)
            #                 stamp_since_last_unknown_image=stamp
            #                 if unknown_count>20:
            #                     print("What's the person's name: ")
            #                     name=input()
            #                     recognitionModel.save_new_face(unknown_images,name)
            #                     unknown_images=[]
            #                     unknown_count=0
            #                     #Constructs the class for recognition model
            #                     recognitionModel =recognition_model(path_to_training_images, training_image_size)
            #                     #Trains the model
            #                     model.train(recognitionModel.training_images, recognitionModel.training_labels)

            # # If the unrecognized images didn't add one to it's list in a while delete them so it doesn't mix random unknown image captures
            # if stamp-stamp_since_last_unknown_image>15:
            #     unknown_images=[]
            #     unknown_count=0

            ############################################
            # Draw stuff                               #
            ############################################

            # Draw trackers 
            for tracker in trackers:
                image_gui = tracker.draw(image_gui)

            # Draw all the detections
            for detection in detections:
                image_gui = detection.draw(image_gui)

            # # Draw all the faces detected
            # for face_detection in face_detections:
            #     image_gui = face_detection.draw(image_gui)

            # Display the results
            cv2.imshow('img', image_gui)
            # Stop if q key is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the VideoCapture object
        cap.release()
        cv2.destroyAllWindows()
        engine.stop()


if __name__ == '__main__':
    main()