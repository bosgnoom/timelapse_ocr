#!/usr/bin/python

"""
    Make a timelapse video from timelapsed AVI files
    Detect timestamp of each frame
    Determine whether to include this frame in the final movie
    Calculate an averaged frame
    Invoke ffmpeg to make an h264 encoded file
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

# Processing data
from itertools import repeat
from mutagen.mp3 import MP3

# For logging
import logging


# Start logger
# Testing different logger methods:
# "Standard" logger:
logging.basicConfig(format='%(levelname)s:%(funcName)s: %(message)s')
logger = logging.getLogger(__name__)

# Multiprocessing logger: (seems to cause double output, at least using PyCharm)
#logger = multiprocessing.log_to_stderr()
#logger = multiprocessing.get_logger(__name__)

# Set loglevel
# TODO: get logging level from argparse
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

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


def analyze_video(video_file, digits, error_folder):
    """
    Load specified video, detect time from each frame
    Return [filename, [frame_number, datetime]]
    """
    logger.info("{}: Analyzing {}...".format(
        multiprocessing.current_process().name, video_file))

    # open video file
    cap = cv2.VideoCapture(video_file)
    logger.debug("{}: Amount of frames in {}: {}".format(
        multiprocessing.current_process().name, video_file, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # Collect the found times here
    timeframes = []

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

                # Skip weekends
                if datum.weekday() < 5:
                    timeframes.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, datum])

            else:
                logger.error("FAILURE IN RECOGNIZING FRAME!")
                filename = "{}/img-{}-{}.png".format(
                    error_folder, os.path.basename(video_file), cv2.CAP_PROP_POS_FRAMES - 1)
                logger.error("Writing to: {}".format(filename))
                cv2.imwrite(filename, frame_snip)

    # close video file
    cap.release()
    return [video_file, len(timeframes), timeframes]


def flatten_timestamps(collection):
    """
    Flatten the collection of video files and timestamps to a list of timestamps
    :param collection: timestamps
    :return: number of datetimes found
    """
    # Double list comprehension... Still trying to comprehense
    times = [frame[1] for video_file in collection for frame in video_file[1]]
    return times


def process_frame(frame, destination_folder):
    """
    Access each video file a second time: get specified frames, calculate averaged frame
    and write image to img folder
    :param frame: [video_file, number_of_frames, [[frame number, timestamp], [...]]
    :param destination_folder: folder where to write to
    :return: True for now... TODO: add return value to check for processing
    """
    logger.info('{}: Processing images from: {}'.format(multiprocessing.current_process().name, frame[0]))

    cap = cv2.VideoCapture(frame[0])
    if cap.isOpened():
        for image in frame[2]:
            if (image[0] >= 1) and (image[0] < cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1):
                logger.debug('Decoding frame number: {}'.format(image[0]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, image[0] - 1)
                ret1, frame1 = cap.read()
                ret2, frame2 = cap.read()
                ret3, frame3 = cap.read()
                frame_result = cv2.addWeighted(frame1, 0.333, frame2, 0.666, 0)
                frame_result = cv2.addWeighted(frame_result, 0.75, frame3, 0.25, 0)
                file_name = "{}/img{}.png".format(destination_folder, image[1].strftime('%Y%m%d%H%M'))
                logger.debug("Writing to: {}".format(file_name))
                cv2.imwrite(file_name, frame_result)

    cap.release()
    return True


def select_timestamps(amount_of_frames_needed, timestamps):
    """
    Use the list of timeframes, select which ones to use
    - The amount of frames needed is e.g. 300 sec * 30 fps = 9000
    - From the list of timestamps it is determined that there are (e.g.) 30 days available
    - So, there are 9000 / 30 = 300 frames per day needed
    - Each frame is 5 minutes apart (from raw footage), so
    - Start time = 12:00 - (300/2) * 5 minutes
    - Stop time = 12:00 + (300/2) * 5 minutes
    
    :param amount_of_frames_needed: the amount of frames needed
    :param timestamps: original list

    :return: list of selected timeframes
    """
    logger.info("Enough frames are available, checking which ones are needed...")

    # Loop over all timestamps, add the days to a list
    # Could be replaced by a double list comprehension, but I am not able to produce this on my own
    days = []
    for video_file in timestamps:
        for time_frame in video_file[2]:
            days.append(time_frame[1].strftime("%Y%m%d"))

    # By converting into a set only unique values remain
    amount_of_days = len(set(days))
    total_frames_per_day = amount_of_frames_needed / amount_of_days

    logger.info("Amount of days in videos found: {}".format(amount_of_days))
    logger.info("Reducing to {:1.1f} frames per day...".format(total_frames_per_day))

    # Calculate start and stop time
    start_time = datetime.timedelta(hours=12) - datetime.timedelta(minutes=5 * (total_frames_per_day / 2))
    stop_time = datetime.timedelta(hours=12) + datetime.timedelta(minutes=5 * (total_frames_per_day / 2))

    logger.info("Selecting frames from {} to {}".format(start_time, stop_time))

    # Make a new list of timestamps
    # [ video_file, number_of_frames, [[frame nr, timestamp], [nr, time], [...]]]
    selected_timestamps = []
    for video_file in timestamps:
        times = []
        for frames in video_file[2]:
            time_frame = frames[1]
            time_to_match = datetime.timedelta(
                hours=time_frame.hour,
                minutes=time_frame.minute,
                seconds=time_frame.second)
            if start_time < time_to_match < stop_time:
                times.append(frames)
        selected_timestamps.append([video_file[0], len(times), times])

    return selected_timestamps


def invoke_ffmpeg(target_fps, music_file, frame_folder, destiny_file):
    """
        Prepare ffmpeg command and execute
        target_fps is the number of frames per second for the movie
        codec based on x264 and aac audio (mobile phone proof settings)
        """
    # First, rename all files to img0xxxx.png to compensate for windows ffmpeg (missing glob)
    file_list = [file for file in os.listdir(frame_folder) if file.endswith('.png')]
    file_list.sort()

    for i, file_name in enumerate(file_list):
        logger.debug("{}-{}".format(i, file_name))
        os.rename(frame_folder + '/' + file_name, frame_folder + '/img{:05d}.png'.format(i))

    command = []
    command.append('ffmpeg')

    # Convert images into video
    command.append("-y -r {} -i {}/img0%4d.png".format(target_fps, frame_folder))

    # Add soundtrack
    command.append('-i {}'.format(music_file))

    # Set video codec
    command.append('-vcodec libx264 -profile:v high -preset slow')
    command.append('-pix_fmt yuv420p')
    command.append('-vprofile baseline -movflags +faststart')
    command.append('-strict -2 -acodec aac -b:a 128k')

    # Cut video/audio stream by the shortest one
    command.append('-shortest')

    # Filename
    command.append('{}'.format(destiny_file))

    command = ' '.join(command)

    result = os.system(str(command))
    logger.debug(result)


def main(folder_name, destiny_file, music_file):
    logger.info("Starting main...")
    logger.info("Processing video folder: {}".format(folder_name))

    # Load the image containing the figures to recognize.
    digits = load_reference_image('cijfers.png')

    # Load the list of video files to process
    raw_material = [avi_file for avi_file in glob.glob('{0}/*/*.AVI'.format(folder_name))]

    # Create a place where to put not recognized time frames
    error_folder = "{}/error".format(folder_name)
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)

    # Recognize the timestamps in the video files
    with multiprocessing.Pool() as pool:
        partial_map = partial(analyze_video, digits=digits, error_folder=error_folder)
        timestamps = pool.map(partial_map, raw_material)

    # check length of timelapse music file, calculate the needed frame rate
    audio_file = MP3(music_file)   # TODO: input from argument
    logger.info("Length of audio file: {} sec".format(audio_file.info.length))

    # Calculate the total amount of frames needed
    target_fps = 30     # TODO: 30 (fps) from argument (or rather maximum frame rate)
    amount_of_frames_needed = target_fps * audio_file.info.length

    # Calculate amount of frames available
    amount_of_frames_available = sum([i[1] for i in timestamps])

    # Calculate the target frames per second
    # Either reduce the amount of frames needed, or lower the frame rate
    if amount_of_frames_needed > amount_of_frames_available:
        # We need more frames than available, so calculate reduced frame rate to fill video
        logger.info("Amount of frames too low. Calculating new frame rate...")
        target_fps = int(amount_of_frames_available / audio_file.info.length + 0.5)
        logger.info('Calculated frame rate: {:0.3f}'.format(target_fps))
    else:
        # There are more frames than needed, reduce the amount of frames
        timestamps = select_timestamps(amount_of_frames_needed, timestamps)

    # If needed create a folder for the processed image files
    frame_folder = "e:/video_tmp".format(folder_name)
    logger.info("Frame folder: {}".format(frame_folder))

    if not os.path.exists(frame_folder):
        logger.debug("Image folder not existing, creating...")
        os.makedirs(frame_folder)
    # Empty the img folder
    for filename in glob.glob("{}/*.png".format(frame_folder)):
        logger.debug("Removing file: {}".format(filename))
        os.remove(filename)

    # Process the selected frames
    with multiprocessing.Pool() as pool:
        # result = pool.starmap(process_frame, zip(timestamps, repeat(frame_folder)))
        partial_map = partial(process_frame, destination_folder=frame_folder)
        result = pool.map(partial_map, timestamps)

        # TODO: check result

    # Invoke ffmpeg
    invoke_ffmpeg(target_fps, music_file, frame_folder, destiny_file)

    logger.info('All done...')


if __name__ == "__main__":
    # If we're started directly, call main() via a callable to measure performance
    # t = timeit.Timer(lambda: main("C:/Users/pauls/Documents/GitHub/timelapse_ocr/video"))
    t = timeit.Timer(lambda: main(
        "E:/Datastore/TLCPRO/XL51", "C:/Users/pauls/Dropbox/Timelapse/2018-03-12-XL51.mp4", "Pong.mp3"))
    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))

    # t = timeit.Timer(lambda: main(
    #    "E:/Datastore/TLCPRO/Grinder", "C:/Users/pauls/Dropbox/Timelapse/2018-03-12-Grinder.mp4", "Gemist.mp3"))
    # print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))

    t = timeit.Timer(lambda: main(
        "E:/Datastore/TLCPRO/FO52", "C:/Users/pauls/Dropbox/Timelapse/2018-03-12-FO52.mp4", "Song_2.mp3"))
    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))

    t = timeit.Timer(lambda: main(
        "E:/Datastore/TLCPRO/Hal_2", "C:/Users/pauls/Dropbox/Timelapse/2018-03-12-Hal_2.mp4", "Ghost.mp3"))
    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))

    t = timeit.Timer(lambda: main(
        "E:/Datastore/TLCPRO/Hal_7b", "C:/Users/pauls/Dropbox/Timelapse/2018-03-12-Hal_7b.mp4", "Tainted_Love.mp3"))
    print("Time needed: {:0.1f} sec".format(t.timeit(number=1)))
