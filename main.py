#!/usr/bin/env python3

import cv2
from detect_faces_functions import detect_faces_Haar


def main():
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # To capture video from webcam. 
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame
        _, img = cap.read()

        #Calls the function that will detect the faces
        detected_faces=detect_faces_Haar(face_cascade, img, scaleFactor = 1.03,minNeighbors=7,minSize=(50, 50))
        

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
