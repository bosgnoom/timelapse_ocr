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
import datetime
#from timeit import Timer
import timeit


def load_reference_image(name):
    """
    Load reference image containing all figures from 0 to 9
    Detect how figures look like
    return dict containing shapes
    """
    print("Loading reference image...")
    # First, load reference image: this image contains the figures 0123456789
    reference = cv2.imread(name)
    reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference = cv2.threshold(reference, 10, 255, cv2.THRESH_BINARY)[1]

    # Find the figures in the reference image
    reference_contours = cv2.findContours(reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # There's a difference in opencv API versions, account for that
    reference_contours = reference_contours[0] if imutils.is_cv2() else reference_contours[1]

    # Sort the found contours from left to right
    reference_contours = contours.sort_contours(reference_contours, method="left-to-right")[0]

    # Put each digit in a dict
    digits={}
    for (i, c) in enumerate(reference_contours):
        print("    Looking for number {}".format(i))
        (x, y, w, h) = cv2.boundingRect(c)
        roi = reference[y:y+h, x:x+w]
        digits[i]=roi

    # Return dict
    return digits


def find_timestamp(frame, digits):
    """
    Detect digits in the frame by applying matchTemplate
    Return the found date/time
    """
    #print("Looking for timeframe...")

    # Cut out the timestamp
    frame = frame[703:720, 560:900]
    
    # Convert frame to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find digits in the image by matchTemplate each digit
    found_numbers = []
    for number, digit in digits.items():
        match_result = cv2.matchTemplate(grey, digit, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        locations = np.where(match_result >= threshold)

        for position in np.transpose(locations):
            found_numbers.append([position[1], number])

    # Sort output from left to right, return only digits, not position, skip (TLC PRO)200
    found_numbers.sort()
    output = [ digit[1] for digit in found_numbers ]

    # Calculate year, month... into a date
    year = 1000*output[0] + 100*output[1] + 10*output[2] + output[3]
    month = 10*output[4] + output[5]
    day = 10*output[6] + output[7]
    hour = 10*output[8] + output[9]
    minute = 10*output[10] + output[11]
    second = 10*output[12] + output[13]
    datum = datetime.datetime(year, month, day, hour, minute, second)

    return datum


def load_video(filename, reference_numbers):
    """
    Load specified video, print time
    """
    print("Loading video {}".format(filename))
    
    # open video file
    cap = cv2.VideoCapture(filename)
    print("Amount of frames: {}".format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Nwe method in progress...
    # Ultimate goal: map(...)
    output=[]
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        cap.set(1, i)
        ret, frame = cap.read()
        tijd = find_timestamp(frame, reference_numbers)
        output.append(tijd)
        print(tijd)

    # old method, for own reference
    """
    # if video file opened, grab a frame
    output = []
    # needs to be optimized: how to access frames from cap.read(). (need to RTFM)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            tijd = find_timestamp(frame, reference_numbers)
            output.append(tijd)
            #print(tijd)
        else:
            break
    """
    
    # close video file
    cap.release()


def main():
    print("Starting main...")
    reference_numbers = load_reference_image('cijfers.png')
    load_video('TLC00032.AVI', reference_numbers)
    print("All done...")

    
if __name__ == "__main__":
    # If we're started directly, call main() via a callable to measure performance
    t = timeit.Timer(lambda: main())
    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))

    """
        Timings:
            - Full frame analysis 34-38 sec
            - Frame cut-out 2-8 sec
    """
