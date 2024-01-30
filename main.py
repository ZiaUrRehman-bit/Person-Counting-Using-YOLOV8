import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import*

model = YOLO('yolov8s.pt')

# read the classes from coco.txt file
myFile = open("coco.txt",'r')
data = myFile.read()
classList = data.split("\n")
print(classList)

cam = cv2.VideoCapture("vidp.mp4")

cTime = 0
pTime = 0

count = 0
personDown = {}
tracker = Tracker()
counter1 = []

personUp = {}
counter2 = []

cy1 = 194
cy2 = 220
offset = 6

emptyFrame = np.ones((480,300,3),np.uint8)*255

# kernel = np.ones((5,5), np.uint8)
while True:
    Success, frame = cam.read()

    copyFrame = frame.copy()

    # grayFrame = cv2.cvtColor(copyFrame, cv2.COLOR_BGR2GRAY)
    # blurFrame = cv2.GaussianBlur(grayFrame,(7,7),0)
    # cannyFrame = cv2.Canny(blurFrame,200,200)

    if not Success:
        break

    count +=1
    if count % 3 != 0:
        continue

    cv2.resize(frame, (1020,500))

    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    # print(px)

    list = []

    for index, row in px.iterrows():
        
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = classList[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])

    bboxId = tracker.update(list)
    for bbox in bboxId:
        x3,y3,x4,y4,id = bbox

        cx = int(x3+x4)//2
        cy = int(y3+y4)//2

        cv2.circle(frame, (cx,cy), 4, (0,255,255), cv2.FILLED)
        cv2.rectangle(frame, (x3,y3),(x4,y4), (0,255,0),2,cv2.FILLED)
        cv2.putText(frame,f"id:{id}",(x3-5,y3),cv2.FONT_HERSHEY_PLAIN,1,(255,70,255),2)
        cv2.putText(emptyFrame,f"id:{id}",(150,150),cv2.FONT_HERSHEY_PLAIN,1,(255,50,255),2)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f"FPS:{int(fps)}", (20,20), cv2.FONT_HERSHEY_COMPLEX,
                1,(0,255,100), 2)

    cv2.imshow("frame", frame)
    cv2.imshow("Empty frame", emptyFrame)
    # cv2.imshow("cannyFrame", cannyFrame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()