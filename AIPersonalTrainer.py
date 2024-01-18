import time
import cv2
import PoseTrackingModule as pm
import numpy as np
#import streamlit as st

#st.title("AI Personal Trainer")

cap = cv2.VideoCapture(0)
#stframe2 = st.empty()
#stframe = st.empty()
detector = pm.poseDetector()
count = 0
dir  = 0
while True:
    success,img =cap.read()
    #img = cv2.resize(img,(1280,720))
    img = detector.findpose(img,False)
    lmList = detector.findPosition(img,False)
    if len(lmList) != 0:
        angle = detector.findAngle(img,12,14,16)
        per = np.interp(angle,(210,310),(0,100))
        bar = np.interp(angle,(220,310),(650,100))
        #print(angle,per)

        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

        #print(count)
        cv2.rectangle(img,(1100,100),(1175,650),(255,0,255),3)
        cv2.rectangle(img,(1100,int(bar)),(1175,650),(255,0,255),cv2.FILLED)
        cv2.putText(img,str(int(count)),(50,100),cv2.FONT_HERSHEY_PLAIN,15,(255,0,255),5)

    #stframe2.text(f"**Count:** {int(count)}")
    #stframe.image(img, channels="img")
    cv2.imshow("Image",img)
    cv2.waitKey(1)
cap.release()