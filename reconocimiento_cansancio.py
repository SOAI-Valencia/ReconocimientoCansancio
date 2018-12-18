from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
from threading import Thread


def sound_alarm(path):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
    #Vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    #Horizontal
    C = dist.euclidean(eye[0], eye[3])
    eye_ratio = (A+B) / (2.0 * C)
    return eye_ratio

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default='./shape_predictor_68_face_landmarks.dat',
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
ap.add_argument("-a", "--alarm", type=str, default="./alarm.wav", help="path alarm .WAV file")
args = vars(ap.parse_args())


#Constans for aspect ratio
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
ALARM_ON = False

COUNTER = 0
EYE_COLOR = (0, 255, 0)

#predictor facil landmarkl
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#get the indexes of the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#start videostream
fileStream = False
if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"]).start()
    fileStream = True

time.sleep(1.0)

while True:

    if fileStream and not vs.more():
        break
    
    frame = vs.read()
    frame = imutils.resize(frame, width = 450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #get the shape
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #get the aspect rati0
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(frame, [leftEyeHull], -1, EYE_COLOR, 1)
        cv2.drawContours(frame, [rightEyeHull], -1, EYE_COLOR, 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.deamon = True
                        t.start()
                cv2.putText(frame, "ALERTA CANSANCIO!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            ALARM_ON = False
        
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()