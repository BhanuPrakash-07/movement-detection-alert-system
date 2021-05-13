import cv2
import pyttsx3
import threading
c=1
def alert(engine):
    global c
    cv2.imwrite(r"B:\Python\ML\MiniProjects\MotionDetection_Alarm/IMG%04i.jpg" %c, frame)
    engine.say("Movement Alert")
    engine.runAndWait()
    c+=1

baseline_image=None
status_list=[None,None]
video=cv2.VideoCapture(0)
engine=pyttsx3.init()
while True:
    check,frame=video.read()
    status=0
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame=cv2.GaussianBlur(gray_frame,(25,25),0)

    if baseline_image is None:
        baseline_image=gray_frame
        continue
    difference=cv2.absdiff(baseline_image,gray_frame)
    threshold=cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]
    (contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour)<10000:
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        t=threading.Thread(target=alert, args=(engine,))
        t.start()
    cv2.imshow('gf',gray_frame)
    cv2.imshow('df',difference)
    cv2.imshow('tf',threshold)
    cv2.imshow("cf",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        engine.stop()
        video.release()
        cv2.destroyAllWindows()
