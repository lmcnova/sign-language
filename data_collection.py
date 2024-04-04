import cv2
import os
from cvzone.HandTrackingModule import HandDetector


name = input("Enter your File name : ")
cap = cv2.VideoCapture("yes.mp4")
dect = HandDetector(maxHands=1)
fr = 0

path = f"dataset/"
os.makedirs(path)
frs = int(cap.get(cv2.CAP_PROP_FPS))
print(frs)

while True:
    try:
        ret, frame = cap.read()
        hand, frame = dect.findHands(frame)
        resize = cv2.resize(frame, (500, 500))
        hand_dec = len(hand)

        if hand_dec != 0:
            ret, frame1 = cap.read()
            cv2.imwrite(os.path.join(f'dataset/{name}/', name + str(fr) + ".jpg"), frame1)
            print(os.path.join(f'dataset2/{name}/', name + str(fr) + ".jpg"))
            fr += 1
            # print(fr)
        # if hand != "":
        #     fr +=1
        #     print(fr)
        cv2.imshow("output", resize)

        if cv2.waitKey(1) == ord('q'):
            break
    except cv2.error:
        print("print", cv2.error)
        break

cap.release()
cv2.destroyAllWindows()















