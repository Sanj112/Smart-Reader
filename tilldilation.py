import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


cap = cv2.VideoCapture(0)

while True:
##    
    ret, frame = cap.read()
    cv2.imshow('ved',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\capturedimage.jpg',frame)
        break
cap.release()
cv2.destroyAllWindows()
from text_region_detection_original import cropimage
capturedimage1=cv2.imread(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\capturedimage.jpg')
cropimage(capturedimage1)
graytext=cv2.cvtColor(capturedimage1,cv2.COLOR_BGR2GRAY)
cv2.imwrite(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\graytext.jpg',graytext)
blur=cv2.GaussianBlur(graytext,(5,5),0)
cv2.imwrite(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\blur.jpg',blur)
threshtext= cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 5)
cv2.imwrite(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\threshtext.jpg',threshtext)
edged = imutils.auto_canny(threshtext)
#dilation
orig=capturedimage1.copy()
kernal = np.ones((1,1), np.uint8)
erosion = cv2.erode(edged, kernal, iterations=2)
opening = cv2.morphologyEx(threshtext, cv2.MORPH_OPEN, kernal)
cv2.imwrite(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\erosion.jpg',erosion)
dilation = cv2.dilate(erosion, kernal, iterations=1)
cv2.imwrite(r'C:\Users\SANJANA\Documents\Sanjana Documents\design project\a\dilation.jpg',dilation)
cv2.imshow('ihduwe',dilation)
##closing = cv2.morphologyEx(threshtext, cv2.MORPH_CLOSE, kernal)
mg = cv2.morphologyEx(threshtext, cv2.MORPH_GRADIENT, kernal)
##th = cv2.morphologyEx(threshtext, cv2.MORPH_TOPHAT, kernal
###Stitles = ['image','mask','maskkk','dilation','erosion','opening','closing','mg','th']
##images = [frame,graytext,threshtext, dilation, erosion, opening, closing, mg, th]
##for i in range(9):
##    plt.subplot(2, 5, i+1), plt.imshow(images[i], 'gray')
##    plt.title(titles[i])
##    plt.xticks([]),plt.yticks([])
##plt.show()
####
##
####contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
##cnts = imutils.grab_contours(contours)
##
### loop over the (unsorted) contours and label them
##for (i, c) in enumerate(cnts):
##	orig = contours.label_contour(orig, c, i, color=(240, 0, 159))
##
### show the original image
##cv2.imshow("Original", orig)
##
### loop over the sorting methods
##for method in ("left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"):
##	# sort the contours
##	(cnts, boundingBoxes) = contours.sort_contours(cnts, method=method)
##	clone = image.copy()
##
##	# loop over the sorted contours and label them
##	for (i, c) in enumerate(cnts):
##		sortedImage = contours.label_contour(clone, c, i, color=(240, 0, 159))
##
##	# show the sorted contour image
##	cv2.imshow(method, sortedImage)
##
### wait for a keypress
##cv2.waitKey(0)

##sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
##
##for i, ctr in enumerate(sorted_ctrs):
##    # Get bounding box
##    x, y, w, h = cv2.boundingRect(ctr)
##roi = dilation[y:y+h, x:x+w]
##cv2.rectangle(dilation,(x,y),( x + w, y + h ),(0,255,0),2)
##if w > 15 and h > 15: 
##        cv2.imwrite('C:\\Users\\Link\\Desktop\\output\\{}.png'.format(i), roi)
##cv2.imshow('marked areas',dilation) 
##cv2.waitKey(0)
