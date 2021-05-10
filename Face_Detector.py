import cv2 
from random import randrange

#Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose an image to detect faces
#img=cv2.imread('pic2.jpg')

webcam=cv2.VideoCapture(0)

#Iterate forever over frames
while True:

    #read current frame
    successful_frame_read,frame=webcam.read()

    #Must convert it to grayscale
    grayscaled_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Detect faces
    #It will give coordinates of rectangle surrounding the face
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangle around faces
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(128,256),randrange(128,256),randrange(128,256)),10)

    cv2.imshow('Face Detector',frame)
    #Pauses the execution of code so that we can view the image
    key=cv2.waitKey(1)

    #Stop if Q key is pressed
    if key==81 or key==113:
        break

#release video capture object
webcam.release()
print('Code completed')