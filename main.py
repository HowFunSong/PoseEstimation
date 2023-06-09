import cv2
# import numpy as np
import time
import mediapipe as mp
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture("video/9.mp4")
pTime = time.time()
while True :
    success , img = cap.read()
    img = cv2.resize(img,(648,960))
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    if results.pose_landmarks :
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id ,lm in enumerate(results.pose_landmarks.landmark) :
            h,w,c = img.shape
            cx,cy = int(lm.x*w) ,int(lm.y*h)
            cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img,f"Fps:{str(int(fps))}",(70,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

