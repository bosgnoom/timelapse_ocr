#!/usr/bin/python

"""
    Extract frames from selected video files, write filename in detected date format
    Detect timestamp of each frame
"""

# Import modules for image analysis
from imutils import contours
import imutils
import numpy as np
import cv2
import datetime

# For measuring and improving performance
import timeit
import multiprocessing
from functools import partial

# File handling
import glob
import os

# Audio processing
from mutagen.mp3 import MP3

# For logging
import logging


# Start logger
# Testing different logger methods:
# "Standard" logger:
logging.basicConfig(format='%(levelname)s:%(funcName)s: %(message)s')
logger = logging.getLogger(__name__)

# Set loglevel
# TODO: get logging level from argparse
#logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

def load_reference_image(name):
    """
    Load reference image containing all figures from 0 to 9
    Detect how figures look like
    return dict containing shapes
    """
    logger.info("Loading reference image...")
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
    digits = {}
    for i, c in enumerate(reference_contours):
        logger.debug("Looking for number {}".format(i))
        x, y, w, h = cv2.boundingRect(c)
        roi = reference[y:y + h, x:x + w]
        digits[i] = roi

    # Return dict
    return digits


def analyze_video(video_file, digits, dest_folder, error_folder):
    """
    Load specified video, detect time from each frame
    """
    logger.info("{}: Analyzing {}...".format(
        multiprocessing.current_process().name, video_file))

    # open video file
    cap = cv2.VideoCapture(video_file)
    logger.debug("{}: Amount of frames in {}: {}".format(
        multiprocessing.current_process().name, video_file, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # if video file opened, grab a frame
    ret = cap.isOpened()
    while ret:
        ret, frame = cap.read()
        if ret:
            # There's a frame decoded, find the timestamp within
            # Look only at the timestamp
            frame_snip = frame[703:720, 560:900]

            # Convert frame to greyscale
            grey = cv2.cvtColor(frame_snip, cv2.COLOR_BGR2GRAY)

            # Find digits in the image by matchTemplate each digit
            found_numbers = []
            for number, digit in digits.items():
                match_result = cv2.matchTemplate(grey, digit, cv2.TM_CCOEFF_NORMED)
                threshold = 0.9
                locations = np.where(match_result >= threshold)

                for position in locations[1]:
                    found_numbers.append([position, number])

            # Sort output from left to right, return only digits, not the position of the digit in the image
            found_numbers.sort()
            numbers = [digit[1] for digit in found_numbers]

            # Process the numbers if there are 14 figures found
            if len(numbers) == 14:
                # Calculate year, month... into a date
                year = 1000 * numbers[0] + 100 * numbers[1] + 10 * numbers[2] + numbers[3]
                month = 10 * numbers[4] + numbers[5]
                day = 10 * numbers[6] + numbers[7]
                hour = 10 * numbers[8] + numbers[9]
                minute = 10 * numbers[10] + numbers[11]
                second = 10 * numbers[12] + numbers[13]
                datum = datetime.datetime(year, month, day, hour, minute, second)

                logger.debug("Found time: {}".format(datum))
                timestamp = "{}".format(datum)
                timestamp = timestamp.replace(" ", "_")
                timestamp = timestamp.replace(":", "-")
                filename = "{}/img-{}.png".format(dest_folder, timestamp)
                cv2.imwrite(filename, frame)
                logger.debug("Filename: {}".format(filename))

            else:
                logger.error("FAILURE IN RECOGNIZING FRAME!")
                filename = "{}/img-{}-{}.png".format(
                    error_folder, os.path.basename(video_file), cv2.CAP_PROP_POS_FRAMES - 1)
                logger.error("Writing to: {}".format(filename))
                cv2.imwrite(filename, frame)

    # close video file
    cap.release()
    return True



def main(folder_name):
    logger.info("Starting main...")
    logger.info("Processing video folder: {}".format(folder_name))

    # Load the image containing the figures to recognize.
    digits = load_reference_image('cijfers.png')

    # Load the list of video files to process
    raw_material = [avi_file for avi_file in glob.glob('{0}/*/*.AVI'.format(folder_name))]

    # Create a place where to put recognized time frames
    error_folder = "{}/error".format(folder_name)
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)

    # Create a place where to put not recognized/error time frames
    dest_folder = "{}/images".format(folder_name)
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Recognize the timestamps in the video files
    with multiprocessing.Pool() as pool:
        partial_map = partial(analyze_video, digits=digits, dest_folder=dest_folder, error_folder=error_folder)
        timestamps = pool.map(partial_map, raw_material)

    logger.info('All done...')


if __name__ == "__main__":
    # If we're started directly, call main() via a callable to measure performance

    t = timeit.Timer(lambda: main(
        "E:/Datastore/TLCPRO/FO52"))

    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))