import cv2
import pyttsx3
import threading
c=1 #variable for image names
def alert(voice):
    global c
    cv2.imwrite(r"B:\Python\ML\MiniProjects\MotionDetection_Alarm/IMG%04i.jpg"%c, frame)      #capturing image when movement detected
    voice.say("Movement Alert")    #giving alert when movement detected
    voice.runAndWait()
    c+=1

initial_image=None   #initial image of the room
status_list=[None,None]  #list of changes in the frames
video=cv2.VideoCapture(0)  #webcam video recording
voice=pyttsx3.init()  #initializing pyttsx3 for text to speech
while True:
    ret,frame=video.read()
    status=0
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame=cv2.GaussianBlur(gray_frame,(25,25),0)    #grayframe for clean image processing

    if initial_image is None:
        initial_image=gray_frame
        continue
    difference=cv2.absdiff(initial_image,gray_frame)   #to show the absolute difference between the frames
    threshold=cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)[1]  #image thresholding
    (contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour)<2000:    #based on room size and disturbances the value can be changed
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        t=threading.Thread(target=alert, args=(voice,))
        t.start()
    cv2.imshow('gf',gray_frame)
    cv2.imshow('df',difference)
    cv2.imshow('tf',threshold)
    cv2.imshow("cf",frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
        voice.stop()
        video.release()
        cv2.destroyAllWindows()
