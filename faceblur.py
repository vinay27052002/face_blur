import cv2 as cv 
import time 

face_cascade=cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
video=cv.VideoCapture(0)

while True:
    check,frame=video.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face=face_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in face:
        img=cv.rectangle(frame,(x,y),(w+x,h+y),(255,255,0),1)
        img[y:y+h,x:x+h]=cv.medianBlur(img[y:y+h,x:x+h],35)

    cv.imshow("blur",frame)
    key=cv.waitKey(1)
    if key==ord("q"):
        break

video.release()
cv.destroyAllWindows()