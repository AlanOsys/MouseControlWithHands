import cv2
import mediapipe as mp
import numpy as np
import autopy

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)


with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
    
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:

                #normalizedLandmarkT = handLandmarks.landmark[4]
                #cv2.circle(image, (round(normalizedLandmarkT.x*640),round(470*normalizedLandmarkT.y)), 8,(0,0,255),-1)
                normalizedLandmarkPI = handLandmarks.landmark[5]
                normalizedLandmarkI = handLandmarks.landmark[8]
                cv2.circle(image, (round(normalizedLandmarkI.x*640),round(470*normalizedLandmarkI.y)), 8,(0,0,255),-1)
                #cv2.circle(image, (round(normalizedLandmarkPI.x*640),round(470*normalizedLandmarkPI.y)), 8,(0,0,255),-1)
                autopy.mouse.move(normalizedLandmarkI.x*1280,940*normalizedLandmarkI.y)
                normalizedLandmarkM = handLandmarks.landmark[12]
                cv2.circle(image, (round(normalizedLandmarkM.x*640),round(470*normalizedLandmarkM.y)), 8,(0,0,255),-1)
                #print(normalizedLandmarkI.x,normalizedLandmarkI.y)
                #print(normalizedLandmarkM.x-normalizedLandmarkI.x)
                if normalizedLandmarkM.x-normalizedLandmarkI.x < 0.03 and normalizedLandmarkM.x-normalizedLandmarkI.x > 0:
                  cv2.circle(image, (round(normalizedLandmarkM.x-normalizedLandmarkI.x*640),round(470*normalizedLandmarkM.y)),8,(255,0,255),-1)
                  autopy.mouse.click()
                
                #normalizedLandmarkR = handLandmarks.landmark[16]
                #cv2.circle(image, (round(normalizedLandmarkR.x*640),round(470*normalizedLandmarkR.y)), 8,(0,0,255),-1)
                #normalizedLandmarkP = handLandmarks.landmark[20]
                #cv2.circle(image, (round(normalizedLandmarkP.x*640),round(470*normalizedLandmarkP.y)), 8,(0,0,255),-1)
                
    
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
