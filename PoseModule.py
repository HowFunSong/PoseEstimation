import cv2
# import numpy as np
import time
import mediapipe as mp
import math

class poseDetector():
    def __init__(self,mode=False,upBody = False,smooth = True,
                 detectionCon = 0.5,trackingCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.upBody,
                                     self.smooth,False,True,self.detectionCon,
                                     self.trackingCon)


    def findPose(self,img, draw=True) :
        # img = cv2.resize(img,(1080,720))
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks :
            if draw :
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
    def findPosition(self,img,draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id ,lm in enumerate(self.results.pose_landmarks.landmark) :
                h,w,c = img.shape
                cx,cy = int(lm.x*w) ,int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw :
                    cv2.circle(img,(cx,cy),5,(255,0,0),cv2.FILLED)
        return self.lmList
    def findAngle(self,img,pos1,pos2,pos3,draw=True):
        #get the lanmarks
        x1 ,y1 =self.lmList[pos1][1:]
        x2 ,y2 =self.lmList[pos2][1:]
        x3 ,y3 =self.lmList[pos3][1:]

        #calculate th angle
        angle = math.degrees(math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2))
        if angle < 0 :
            angle += 180
        elif angle > 180:
            angle = 360 - angle

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),3)
            cv2.line(img,(x2,y2),(x3,y3),(0,0,255),3)
            cv2.circle(img,(x1,y1),5,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),10,(0,0,255),2)
            cv2.circle(img,(x2,y2),5,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(0,0,255),2)
            cv2.circle(img,(x3,y3),5,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),10,(0,0,255),2)
            cv2.putText(img, f"{angle:.1f}", (x2+20, y2-20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture("video/9.mp4")
    pTime = time.time()
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        # print(lmList)
        # cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (0, 255, 255), cv2.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"Fps:{str(int(fps))}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__" :
    main()

