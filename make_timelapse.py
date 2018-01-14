#!/usr/bin/python

"""
    Make a timelapse video from timelapsed AVI files
    Detect timestamp of each frame
    Determine whether to include this frame in the final movie
    Calculate an averaged frame
    Invoke ffmpeg to make an h264 encoded file
"""

# Import modules
from imutils import contours
import imutils
import numpy as np
import cv2

def load_reference_image(name):
    # First, load reference image: this image contains the figures 0123456789
    reference = cv2.imread(name)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference = cv2.threshold(reference, 10, 255, cv2.THRESH_BINARY)[1]

    # Find the figures in the reference image
    reference_contours = cv2.findContours(reference.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # There's a difference in opencv API versions, account for that
    reference_contours = reference_contours[0] if imutils.is_cv2() else reference_contours[1]
    reference_contours = contours.sort_contours(reference_contours, method="left-to-right")[0]

    # Put each digit in a dict
    digits={}
    for (i, c) in enumerate(reference_contours):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = reference[y:y+h, x:x+w]
        digits[i]=roi

    # Return dict
    return digits

def find_timestamp(digits):
    # open video file
    cap = cv2.VideoCapture('TLC00032.AVI')

    # if video file opened, grab a frame
    if cap.isOpened():
        ret, frame = cap.read()
        # Convert to greyscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find digits in the image by matchTemplate each digit
        output=[]
        for number, digit in digits.items():
            result = cv2.matchTemplate(gray, digit, cv2.TM_CCOEFF_NORMED)
            threshold = 0.9
            locations = np.where(result >= threshold)

            for position in np.transpose(locations):
                if position[0] == 704:
                    output.append([number, position[1]])    
                      
    # close video file
    cap.release()

    print(output)

def main():
    print("Starting...")
    numbers = load_reference_image('cijfers.png')
    find_timestamp(numbers)

if __name__ == "__main__":
    main()
    
