import cv2
import time

cam = cv2.VideoCapture("vidp.mp4")

cTime = 0
pTime = 0

while True:
    Success, frame = cam.read()

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