import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image  
from pytesseract import image_to_string
import RPi.GPIO as GPIO
import time
import cv2
import gtts

GPIO.setmode(GPIO.BOARD)

GPIO.setup(36, GPIO.IN, pull_up_down=GPIO.PUD_UP)
cap = cv2.VideoCapture(0)
while True:
    ret,frame =cap.read()
    cv2.imshow('ved',frame)
    input_state = GPIO.input(36)
    if cv2.waitKey(1) & input_state == False:
        cv2.imwrite('LOCATION TO BE SAVED TO',frame)
        break
cap.release()
cv2.destroyAllWindows()


from text_region_detection import cropimage#importing another program for cropping image 
capturedimage1=cv2.imread('LOCATION OF CAPTURED IMAGE')
cropimage(capturedimage1)
graytext=cv2.cvtColor(capturedimage1,cv2.COLOR_BGR2GRAY)
cv2.imwrite('LOCATION TO BE SAVED TO',graytext)
blur=cv2.GaussianBlur(graytext,(5,5),0)
cv2.imwrite('LOCATION TO BE SAVED TO',blur)
threshtext= cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 5)
cv2.imwrite('LOCATION TO BE SAVED TO',threshtext)
orig=capturedimage1.copy()
kernal = np.ones((1,1), np.uint8)
erosion = cv2.erode(threshtext, kernal, iterations=2)
opening = cv2.morphologyEx(threshtext, cv2.MORPH_OPEN, kernal)
cv2.imwrite('LOCATION TO BE SAVED TO',erosion)
dilation = cv2.dilate(erosion, kernal, iterations=1)
cv2.imwrite('LOCATION TO BE SAVED TO',dilation)
closing = cv2.morphologyEx(threshtext, cv2.MORPH_CLOSE, kernal)
mg = cv2.morphologyEx(threshtext, cv2.MORPH_GRADIENT, kernal)
edges = cv2.Canny(graytext,50,150,3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for r,theta in lines[0]:
    a=np.cos(theta)
    b=np.sin(theta)
    x0 = a*r
    y0 = b*r
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(-b))
    capt=cv2.line(capturedimage1,(x1,y1), (x2,y2), (255,255,255),2)
cv2.imwrite('LOCATION TO BE SAVED TO',capt)
img =Image.open('LOCATION OF THRESHOLD IMAGE')
n=image_to_string(img)
text_file = open("texts.txt","w+")
text_file.write(n)
text_file.close()
blabla = ("my voice")
myfile = open("texts.txt", "rt")
contents = myfile.read()         
myfile.close()                       
tts = gtts.gTTS(text=contents, lang='en')
tts.save("rec.mp3")
os.system('mpg321 rec.mp3 &')#THIS PLAYS AUDIO

