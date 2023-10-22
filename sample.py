import cv2
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.video import FPS
import dlib
import numpy as np
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from scipy.spatial import distance
import matplotlib.pyplot as plt 

# Initialize Firebase Admin SDK with your credentials JSON file
cred = credentials.Certificate("service.json")
firebase_admin.initialize_app(cred)

# Initialize the Firestore client
db = firestore.client()

# Define a reference to the Firestore collection where you want to insert data
collection_ref = db.collection('cvdata')

# Data to insert (in dictionary format)



# Load your image
image = cv2.imread("image2.jpg")

# Initialize dlib's face detector (assuming you have dlib's shape_predictor_68_face_landmarks.dat file)
detector = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# Define some constants for your eye detection and alert logic
lStart, lEnd = 42, 48
rStart, rEnd = 36, 42
thresh = 0.25
frame_check = 20
flag = 0
data_to_insert={}
while True:
    # Perform your image processing here
    frame = imutils.resize(image, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detector(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                data_to_insert = {
                    'sleep': True,
                    'user': 'User 1'
                }
                print("sleeping")
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            flag = 0
            print("awake")
            data_to_insert = {
                'sleep': False,
                'user': 'User 1'
            }
    collection_ref.add(data_to_insert)
    # Display the frame using matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.show()
    # cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cv2.destroyAllWindows()
        break
