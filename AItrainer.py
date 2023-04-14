import cv2
import numpy as np
import mediapipe as mp
import time
import PoseModule as pm


cap = cv2.VideoCapture("video/11.mp4")
detector = pm.poseDetector()
count = 0
dir = 0
pTime = 0
while True :
    success , img = cap.read()
    img = cv2.resize(img, (540, 1024))
    # img = cv2.imread("video/test.jpg")
    img = detector.findPose(img,draw = False)
    lmList = detector.findPosition(img,draw = False)
    # print(lmList)
    if len(lmList) != 0 :
        #left side
        detector.findAngle(img,23,25,27)
        #right side
        angle = detector.findAngle(img,24,26,28)
        per = np.interp(angle,(70,180),(100,0))
        bar = np.interp(angle,(70,180),(0,150))
        print(bar)

        #check the squart times
        if int(per) == 20:
            if dir == 0 :
                count += 0.5
                dir = 1
        if int(per) == 80:
            if dir == 1:
                count += 0.5
                dir = 0
        # print(int(count))
        if 67.5 > per > 50 :
            color = (0,255,255)
        elif  85 > per >= 67.5 :
            color = (0,135,255)
        elif per >= 85  :
            color = (0,0,255)
        else :
            color = (125,255,0)
        cv2.rectangle(img,(470,800),(520,950),(0,0,0),4)
        cv2.rectangle(img,(473,800+int(bar)),(517,947),color,cv2.FILLED)
        cv2.putText(img,f"{per:.1f}%",(430,770),cv2.FONT_HERSHEY_PLAIN,2,color,3)

        # cv2.rectangle(img,(0,900),(150,1050),(0,255,0),cv2.FILLED)
        # cv2.putText(img,str(int(count)),(20,1010),cv2.FONT_HERSHEY_PLAIN,10,(0,0,0),10)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.imshow("Image",img)
    cv2.waitKey(0)
