from imutils import contours
import numpy as np
#import argparse
import imutils
import cv2

#load reference image
ref = cv2.imread('cijfers.png')
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY)[1]

#cv2.imshow("ref", ref)

# Find the figures in the reference image
refcnts=cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refcnts=refcnts[0] if imutils.is_cv2() else refcnts[1]
refcnts=contours.sort_contours(refcnts, method="left-to-right")[0]
digits={}

# Put each digit in a dict
for (i, c) in enumerate(refcnts):
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y+h, x:x+w]
    #cv2.imshow(" roi", roi)
    #cv2.waitKey(0)
    digits[i]=roi

#print(digits)
#rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))

#image = cv2.imread('cijfers.png')
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("grijs", gray)

# open video file
cap = cv2.VideoCapture('TLC00032.AVI')

# if video file opened, grab a frame
if cap.isOpened():
    ret, frame = cap.read()
    #cv2.imshow('rgb', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)

    output=[]
    for number, digit in digits.items():
        res = cv2.matchTemplate(gray, digit, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(res >= threshold)

        #print(loc[1])
        for location in loc[1]:
            output.append([location, number])
        for pt in zip(*loc[::-1]):
            cv2.rectangle(frame, pt, (pt[0]+w, pt[1]+h), (0,255,0), 1)
                      
    cv2.imshow('frame', frame)                  
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print(output)
