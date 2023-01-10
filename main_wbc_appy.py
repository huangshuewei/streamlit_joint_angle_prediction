# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 00:06:46 2023

@author: ASUS
"""
import cv2
import streamlit as st
from keras import models
import mediapipe as mp
import numpy as np
import math
#from streamlit_joint_angle_prediction import grade_prediction
#import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

###
@st.cache(ttl=24*3600)
def angles_calculation(landmark_0, landmark_1, landmark_2):
    
    ba = np.array(landmark_0) - np.array(landmark_1)
    bc = np.array(landmark_2) - np.array(landmark_1)
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))  
    angle = np.arccos(cosine_angle)
    angle = round(math.degrees(angle))
    angle = 180 - angle

    return angle

@st.cache(ttl=24*3600)
def grade_prediction(landmarks_3d):
    # load model
    model = models.load_model("NN_model_1.h5",compile=False)
    
    idx = []
    mdl = []
    rng = []
    pky = []
    data = []
    # calculate joint angles
    # index
    idx_dip = angles_calculation(landmarks_3d[8], landmarks_3d[7], landmarks_3d[6])
    idx_pip = angles_calculation(landmarks_3d[7], landmarks_3d[6], landmarks_3d[5])
    idx_mcp = angles_calculation(landmarks_3d[6], landmarks_3d[5], landmarks_3d[0])
    idx.append(idx_dip)
    idx.append(idx_pip)
    idx.append(idx_mcp)
    
    # middle
    mdl_dip = angles_calculation(landmarks_3d[12], landmarks_3d[11], landmarks_3d[10])
    mdl_pip = angles_calculation(landmarks_3d[11], landmarks_3d[10], landmarks_3d[9])
    mdl_mcp = angles_calculation(landmarks_3d[10], landmarks_3d[9], landmarks_3d[0])
    mdl.append(mdl_dip)
    mdl.append(mdl_pip)
    mdl.append(mdl_mcp)
    
    # ring
    rng_dip = angles_calculation(landmarks_3d[16], landmarks_3d[15], landmarks_3d[14])
    rng_pip = angles_calculation(landmarks_3d[15], landmarks_3d[14], landmarks_3d[13])
    rng_mcp = angles_calculation(landmarks_3d[14], landmarks_3d[13], landmarks_3d[0])
    rng.append(rng_dip)
    rng.append(rng_pip)
    rng.append(rng_mcp)
    
    # pinky
    pky_dip = angles_calculation(landmarks_3d[20], landmarks_3d[19], landmarks_3d[18])
    pky_pip = angles_calculation(landmarks_3d[19], landmarks_3d[18], landmarks_3d[17])
    pky_mcp = angles_calculation(landmarks_3d[18], landmarks_3d[17], landmarks_3d[0])
    pky.append(pky_dip)
    pky.append(pky_pip)
    pky.append(pky_mcp)
    
    data.append(idx)
    data.append(mdl)
    data.append(rng)
    data.append(pky)
    data = np.array(data)
    grades = model.predict(data)
    grades = np.argmax(grades,axis = 1)
    
    return grades + 1

@st.cache(ttl=24*3600)
def my_putText(frame, grades):
    
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
    
    return frame

@st.cache(ttl=24*3600)
def img_process(frame):
    
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
        grades = grade_prediction(landmarks_3d)
        frame = my_putText(frame, grades)
    else:
        frame = frame
        
    return frame
    
class VideoProcessor:
    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        img = img_process(frame)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

###

# MediaPipe Hand
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False
                      ,max_num_hands=1,
                      min_detection_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

st.title("Webcam Live Feed")
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# if __name__ == "__main__":
# main()
