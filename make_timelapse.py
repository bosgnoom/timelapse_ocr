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
from itertools import repeat
from mutagen.mp3 import MP3
import logging

# Start logger
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(funcName)s: %(message)s')
# wow... as global variable? I thought this was a no-no...
logger = logging.getLogger(__name__)


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


def find_timestamp(frame, digits):
    """
    Detect digits in the frame by applying matchTemplate
    Return the found date/time
    """
    logger.debug("Looking for timeframe...")

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

    logger.debug("Found time: {}".format(datum))
    return datum


def analyze_video(filename, reference_numbers):
    """
    Load specified video, detect time from each frame
    Return [filename, [frame_number, datetime]]
    """
    logger.info("{}: Analyzing {}...".format(
        multiprocessing.current_process().name, filename))

    # open video file
    cap = cv2.VideoCapture(filename)
    logger.info("{}: Amount of frames in {}: {}".format(
        multiprocessing.current_process().name, filename, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # if video file opened, grab a frame
    output = []
    ret = cap.isOpened()
    while ret:
        ret, frame = cap.read()
        if ret:
            # There's a frame decoded
            epoch = find_timestamp(frame, reference_numbers)
            # Skip weekends
            if epoch.weekday() < 5:
                output.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, epoch])

    # close video file
    cap.release()
    return [filename, output]


def fetch_timestamps(collection):
    """
    Flatten the collection of video files and timestamps to a list of timestamps
    :param collection: timestamps
    :return: number of datetimes found
    """
    # Double list comprehension... Still trying to comprehense
    times = [frame[1] for video_file in collection for frame in video_file[1]]
    return times


def process_frame(frame):
    logger.debug('Processing images from: {}'.format(frame[0]))
    cap = cv2.VideoCapture('TLC00001.AVI')
    if cap.isOpened():
        for image in frame[1]:
            if (image[0] >= 1) and (image[0] < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1):
                logger.debug('Decoding frame: {}'.format(image[0]))
                cap.set(1, image[0] - 1)
                ret1, frame1 = cap.read()
                ret2, frame2 = cap.read()
                ret3, frame3 = cap.read()
                frame_result = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
                frame_result = cv2.addWeighted(frame_result, 0.666, frame3, 0.333, 0)
                file_name = "c:/datastore/tmp/img{}.png".format(image[1].strftime('%Y%m%d%H%M'))
                logger.debug("Writing to: {}".format(file_name))
                cv2.imwrite(file_name, frame_result)

    cap.release()
    return True


def main():
    logger.info("Starting main...")

    # Load the image containing the figures to recognize.
    reference_numbers = load_reference_image('cijfers.png')

    # Load the list of video files to process
    raw_material = [avi_file for avi_file in glob.glob("*.AVI")]     # TODO: input from argument

    # Recognize the timestamps in the video files
    with multiprocessing.Pool(processes=4) as pool:
        timestamps = pool.starmap(analyze_video, zip(raw_material, repeat(reference_numbers)))

    # check length of time lapse music file, calculate the needed frame rate
    audio_file = MP3('Housewife.mp3')   # TODO: input from argument
    logger.info("Length of audio file: {} sec".format(audio_file.info.length))

    # Calculate the total amount of frames needed
    target_fps = 30     # TODO: 30 (fps) from argument (or rather maximum frame rate)
    amount_of_frames_needed = target_fps * audio_file.info.length
    amount_of_frames_needed = 100   # TODO: remove this, just for testing...

    # Calculate the target_fps
    # Either reduce the amount of frames needed, or lower the frame rate
    all_timestamps = fetch_timestamps(timestamps)
    if amount_of_frames_needed > len(all_timestamps):
        # We need more frames than available, so calculate reduced frame rate to fill video
        logger.info("Amount of frames too low. Calculating new frame rate...")
        target_fps = len(all_timestamps) / audio_file.info.length
        logger.info('Calculated frame rate: {}'.format(target_fps))
    else:
        # There are more frames than needed, reduce the amount of frames
        logger.info("Enough frames, checking which ones are needed...")
        days = [day.strftime("%Y%m%d") for day in all_timestamps]
        # By converting into a set only unique values remain
        amount_of_days = len(set(days))
        total_frames_per_day = len(all_timestamps) / amount_of_days

        logger.info("Amount of days in videos found: {}".format(amount_of_days))
        logger.info("Reducing to {:1.1f} frames per day...".format(total_frames_per_day))

        start_time = datetime.timedelta(hours=12) - datetime.timedelta(minutes=5 * (total_frames_per_day / 2))
        stop_time = datetime.timedelta(hours=12) + datetime.timedelta(minutes=5 * (total_frames_per_day / 2))

        logger.info("Selecting frames from {} to {}".format(start_time, stop_time))
        selected_timestamps = []
        for video_file in timestamps:
            times = []
            for frames in video_file[1]:
                time_frame = frames[1]
                time_to_match = datetime.timedelta(
                        hours=time_frame.hour,
                        minutes=time_frame.minute,
                        seconds=time_frame.second)
                if start_time < time_to_match < stop_time:
                    times.append(frames)
            selected_timestamps.append([video_file[0], times])
        timestamps = selected_timestamps

    # TODO: calculate averaged frame in
    # result = [process_frame(frame) for frame in timestamps
    process_frame(timestamps[0])

    logger.info('All done...')


if __name__ == "__main__":
    # If we're started directly, call main() via a callable to measure performance
    t = timeit.Timer(lambda: main())
    logger.info("Time needed: {:0.1f} sec".format(t.timeit(number=1)))
