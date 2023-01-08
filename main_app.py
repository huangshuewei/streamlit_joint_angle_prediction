# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 00:06:46 2023

@author: ASUS
"""
import cv2
import streamlit as st
from keras import models
import mediapipe as mp
from streamlit_joint_angle_prediction import grade_prediction
import time

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# load model
model = models.load_model("NN_model_1.h5",compile=False)

# for calculate FPS
previousTime = 0
currentTime = 0

# MediaPipe Hand
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False
                      ,max_num_hands=1,
                      min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# web layout
# c1, c2 = st.columns(2)
camera = cv2.VideoCapture(0)
#with c1:
while(run):
    _, frame = camera.read()
    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # do peocess MediaPipe Hand
    results = hands.process(imageRGB)
    landmarks_3d = []
    
    # collect landmarks
    if results.multi_hand_world_landmarks:
        for handLms in results.multi_hand_world_landmarks: # working with each hand
            for id, lm in enumerate(handLms.landmark):
                cx, cy, cz = lm.x, lm.y, lm.z
                landmarks_3d.append([cx, cy, cz])
        
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks: # working with each hand
            
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    
    if len(landmarks_3d) == 21:
        # grade prediction
        grades = grade_prediction(landmarks_3d,model)
        
        frame = cv2.putText(frame,
                            "Idx: " + str(grades[0]), 
                            (20,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,
                            (0,0,0),
                            1,
                            cv2.LINE_AA)
        frame = cv2.putText(frame,
                            "Mdl: " + str(grades[1]), 
                            (20,90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,
                            (0,0,0),
                            1,
                            cv2.LINE_AA)
        frame = cv2.putText(frame,
                            "Rng: " + str(grades[2]), 
                            (20,120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,
                            (0,0,0),
                            1,
                            cv2.LINE_AA)
        frame = cv2.putText(frame,
                            "Pky: " + str(grades[3]), 
                            (20,150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,
                            (0,0,0),
                            1,
                            cv2.LINE_AA)
        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime-previousTime)
        previousTime = currentTime
        
        frame = cv2.putText(frame, str(int(fps))+" FPS", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 1)
    else:
        frame = frame
    
    FRAME_WINDOW.image(frame)
else:
    camera.release()
    # st.write(camera.isOpened())

#with c2:






