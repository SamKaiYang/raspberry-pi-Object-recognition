import cv2
import time
import os
import glob as gb
cap = cv2.VideoCapture(0)


while(True):
    #cap = cv2.VideoCapture(0)
    if cap.isOpened(): #判断是否正常打开
        ret, frame = cap.read()
    else:
        ret = False
        break

    # 將圖片轉為灰階
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):#擷取圖片
    
        cap.set(3,320)
        cap.set(4,240)
        cap.set(1, 10.0)


        #ret, frame = cap.read()
        fileName = time.strftime("img_test\\"'%Y_%m_%d_%H_%M_%S')+".jpg"
        cv2.imwrite(fileName,frame)
        print(fileName)
        cap = cv2.VideoCapture(0)
    
    if cv2.waitKey(1) & 0xFF == ord('p'):#關閉攝影機
        break

cap.release()
cv2.destroyAllWindows()