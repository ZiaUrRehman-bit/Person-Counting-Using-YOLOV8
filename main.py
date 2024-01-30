import cv2
import time
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

while True:
    Success, frame = cam.read()
    if not Success:
        break

    

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f"FPS:{int(fps)}", (20,20), cv2.FONT_HERSHEY_COMPLEX,
                1,(0,255,100), 2)

    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()