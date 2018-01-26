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
import timeit
import glob
import multiprocessing
import os
from mutagen.mp3 import MP3


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
    digits = {}
    for (i, c) in enumerate(reference_contours):
        # print("    Looking for number {}".format(i))
        x, y, w, h = cv2.boundingRect(c)
        roi = reference[y:y + h, x:x + w]
        digits[i] = roi

    # Return dict
    return digits


def find_timestamp(frame, digits):
    """
    Detect digits in the frame by applying matchTemplate
    Return the found date/time
    """
    # print("Looking for timeframe...")

    # Look only at the timestamp
    frame = frame[703:720, 560:900]

    # Convert frame to greyscale
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
    output = [digit[1] for digit in found_numbers]

    # Calculate year, month... into a date
    year = 1000 * output[0] + 100 * output[1] + 10 * output[2] + output[3]
    month = 10 * output[4] + output[5]
    day = 10 * output[6] + output[7]
    hour = 10 * output[8] + output[9]
    minute = 10 * output[10] + output[11]
    second = 10 * output[12] + output[13]
    datum = datetime.datetime(year, month, day, hour, minute, second)

    return datum


def analyze_video(filename, reference_numbers):
    """
    Load specified video, detect time from each frame
    Return [filename, frame_number, datetime]
    """
    print("{}: Analyzing {}...".format(
        multiprocessing.current_process().name, filename))

    # open video file
    cap = cv2.VideoCapture(filename)
    print("{}: Amount of frames in {}: {}".format(
        multiprocessing.current_process().name, filename, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # if video file opened, grab a frame
    output = []
    ret = cap.isOpened()
    while ret:
        ret, frame = cap.read()
        if ret:
            tijd = find_timestamp(frame, reference_numbers)
            # iets = tijd.strftime("img_%Y-%m-%d-%H%M%S.png")
            # cv2.imwrite("c:\datastore\hal_2\{}".format(iets), frame)
            # print("Tijdstip: {}".format(iets))
            output.append([filename,
                           int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                           tijd])

    # close video file
    cap.release()
    return output


def no_weekend(time_of_day):
    """
    Check what day of the week the time_of_day is
    Return true for weekdays, false for weekends
    """

    return time_of_day.weekday() < 5


def count_the_days(collection):
    days = [day[2].strftime("%Y%m%d") for day in collection]
    return len(set(days))


def is_this_frame_needed(time_frame, start_time, stop_time):
    """
    Check if time_frame is within start and stop time
    :param time_frame: which timestamp to check
    :param start_time: starting time
    :param stop_time:  end time
    :return: true/false
    """
    time_to_match = datetime.timedelta(hours=time_frame.hour, minutes=time_frame.minute, seconds=time_frame.second)
    return_value = start_time < time_to_match < stop_time
    # print("{}: {}".format(time_frame, return_value))
    return return_value


def main():
    print("Starting main...")

    # Load the image containing the figures to recognize.
    reference_numbers = load_reference_image('cijfers.png')

    # Load the list of video files to process
    raw_material = [[x, reference_numbers] for x in glob.glob("*.AVI")]     # TODO: input from argument

    # Recognize the timestamps in the video files
    with multiprocessing.Pool(processes=4) as pool:
        timestamps = pool.starmap(analyze_video, raw_material)

    # Flatten the results in timestamps
    timestamps = [entry for sublist in timestamps for entry in sublist]

    # check length of timelapse music file, calculate the needed frame rate
    audio_file = MP3('Housewife.mp3')   # TODO: input from argument
    print("Length of audio file: {} sec".format(audio_file.info.length))

    # Skip weekends
    timestamps = [frame for frame in timestamps if no_weekend(frame[2])]
    print("Amount of frames found: {}".format(len(timestamps)))

    # Calculate the total amount of frames needed
    target_fps = 30     # TODO: 30 (fps) from argument
    amount_of_frames_needed = target_fps * audio_file.info.length
    amount_of_frames_needed = 100 # TODO: remove this, just for testing...

    # Either reduce the amount of frames needed, or lower the frame rate
    if amount_of_frames_needed > len(timestamps):
        # We need more frames than available, so calculate reduced frame rate to fill video
        print("Amount of frames too low. Calculating new frame rate...")
        target_fps = len(timestamps) / audio_file.info.length
        print('Calculated frame rate: {}'.format(target_fps))
    else:
        # There are more frames than needed, reduce the amount of frames
        amount_of_days = count_the_days(timestamps)
        total_frames_per_day = len(timestamps) / amount_of_days

        print("Amount of days in videos found: {}".format(amount_of_days))
        print("Reducing to {:1.1f} frames per day...".format(total_frames_per_day))

        start_time = datetime.timedelta(hours=12) - datetime.timedelta(minutes=5 * (total_frames_per_day / 2))
        stop_time = datetime.timedelta(hours=12) + datetime.timedelta(minutes=5 * (total_frames_per_day / 2))

        timestamps = [frame for frame in timestamps if is_this_frame_needed(frame[2], start_time, stop_time)]

    # TODO: calculate averaged frame in
    # result = [process_frame(frame) for frame in timestamps

    print('All done...')


if __name__ == "__main__":
    # If we're started directly, call main() via a callable to measure performance
    t = timeit.Timer(lambda: main())
    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))
