#!/usr/bin/python

"""
    Make a timelapse video from source AVI files:
    - Detect timestamp of each frame
    - Determine whether to include this frame in the final movie
    - Calculate averaged frames
    - Invoke ffmpeg to make a x264 encoded file
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

# Parse arguments
import argparse

# Start logger
logging.basicConfig(format='[%(levelname)s/%(funcName)s] %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
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


def determine_timestamps(video_file, digits, error_folder):
    """
    Load specified video, detect timestamps for each frame
    Return [filename, [frame_number, datetime]]
    """
    logger.info("Analyzing {}...".format(video_file))

    # open video file
    cap = cv2.VideoCapture(video_file)
    logger.debug("Amount of frames in {}: {}".format(
        video_file, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # Collect the found times here
    timeframes = []

    # if video file opened, grab a frame
    ret = cap.isOpened()
    while ret:
        ret, frame = cap.read()
        # print("file: {} - ret: {}".format(video_file, ret))
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
            # print("Found numbers: {}-{}".format(video_file, numbers))
            if len(numbers) == 14:
                # Calculate year, month... into a date
                year = 1000 * numbers[0] + 100 * numbers[1] + 10 * numbers[2] + numbers[3]
                month = 10 * numbers[4] + numbers[5]
                day = 10 * numbers[6] + numbers[7]
                hour = 10 * numbers[8] + numbers[9]
                minute = 10 * numbers[10] + numbers[11]
                second = 10 * numbers[12] + numbers[13]
                datum = datetime.datetime(year, month, day, hour, minute, second)

                # logger.debug("Found time: {}".format(datum))

                # Skip weekends
                if datum.weekday() < 5:
                    timeframes.append([int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1, datum])
                else:
                    # print("Video {} is from a weekend...".format(video_file))
                    pass

            else:
                logger.error("FAILURE IN RECOGNIZING FRAME!")
                filename = "{}/img-{}-{}.png".format(
                    error_folder, os.path.basename(video_file), cv2.CAP_PROP_POS_FRAMES - 1)
                logger.error("Writing to: {}".format(filename))
                cv2.imwrite(filename, frame_snip)

    # close video file
    cap.release()
    # print("Return from determine_timestamps: {}".format([video_file, len(timeframes)]))
    return [video_file, len(timeframes), timeframes]


def select_timestamps(amount_of_frames_needed, timestamps):
    """
    Use the list of timeframes, select which ones to use
    - The amount of frames needed is e.g. 300 sec * 30 fps = 9000
    - From the list of timestamps it is determined that there are (e.g.) 30 days available
    - So, there are 9000 / 30 = 300 frames per day needed
    - Each frame is 5 minutes apart (assumption from raw footage), so
    - Start time = 12:00 - (300/2) * 5 minutes
    - Stop time = 12:00 + (300/2) * 5 minutes
    
    :param amount_of_frames_needed: the amount of frames needed
    :param timestamps: original list

    :return: list of selected timeframes
    """
    logger.debug("Enough frames are available, checking which ones are needed...")

    # Loop over all timestamps, add the days to a list
    # Could be replaced by a double list comprehension, but I am not able to produce this on my own
    # Or rather, a double list comprehension is not comprehensible for me ;-)
    days = []
    for video_file in timestamps:
        for time_frame in video_file[2]:
            days.append(time_frame[1].strftime("%Y%m%d"))

    # By converting into a set only unique values remain
    amount_of_days = len(set(days))
    total_frames_per_day = amount_of_frames_needed / amount_of_days

    logger.debug("Amount of days in videos found: {}".format(amount_of_days))
    logger.debug("Reducing to {:1.1f} frames per day...".format(total_frames_per_day))

    # Calculate start and stop time
    start_time = datetime.timedelta(hours=12) - datetime.timedelta(minutes=5 * (total_frames_per_day / 2.0) + 2.5)
    stop_time = datetime.timedelta(hours=12) + datetime.timedelta(minutes=5 * (total_frames_per_day / 2.0) + 2.5)

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

    return [selected_timestamps, start_time, stop_time]


def cleanup_image_folder(image_folder, start_time, stop_time):
    """
    Check image folder, to be sure only the needed images are left
    :param start_time: start time of needed image
    :param stop_time: end time of needed image
    :param image_folder: image containing images
    :return: nothing
    """

    if start_time and stop_time:
        logger.debug("Frame folder might need cleaning up")
        images = glob.glob("{}/img*.png".format(image_folder))
        for image in images:
            path, image = os.path.split(image)
            try:
                image_time = datetime.datetime.strptime(image, "img%Y%m%d%H%M.png")
                if start_time < datetime.timedelta(hours=image_time.hour,
                    minutes=image_time.minute) < stop_time:
                    # logger.debug("Keeping {}".format(image))
                    # No sure how to invert this logic. Readability counts...
                    pass
                else:
                    logger.debug("Removing {}/{}".format(path, image))
                    try:
                        os.remove("{}/{}".format(path, image))
                    except OSError:
                        logger.error("Cannot remove {}/{}...".format(path, image))
            except ValueError:
                logger.error("Cannot parse date from {}/{}...".format(path, image))

    return True


def process_frames(frame, destination_folder):
    """
    Access each video file a second time: get specified frames, calculate averaged frame
    and write image to img folder
    :param frame: [video_file, number_of_frames, [[frame number, timestamp], [...]]
    :param destination_folder: folder where to write to
    :return: true if all frames are written to disk
    """
    return_value = True
    logger.info('Processing images from: {}'.format(frame[0]))
    # print(frame)

    cap = cv2.VideoCapture(frame[0])

    # Cache small files to increase processing speed.
    # Access large files from disk to prevent out-of-memory faults
    cache = []
    if os.path.getsize(frame[0]) < 25000000:    # Let's start with 25 mb
        logger.debug("Caching video file {}...".format(frame[0]))
        ret = cap.isOpened()
        # print("Ret: {}-{}".format(frame[0], ret))
        while ret:
            ret, image = cap.read()
            if ret:
                cache.append(image)
    else:
        logger.debug("Video file {} is too large, direct access method chosen...".format(frame[0]))
        pass

    # Process each image in the list of frames.
    # Check if image already exists (means lower processing time is needed)
    # Get two frames and calculate the averaged frame
    # Write back as file
    for image in frame[2]:
        image_name = image[1].strftime('%Y%m%d%H%M')
        file_name = "{}/img{}.png".format(destination_folder, image_name)
        if cache:
            frame_count = len(cache)
        else:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if image[0] >= 1 and not os.path.exists(file_name):    # assure that we can access all frames
            logger.debug('Decoding frame number: {}/{} from {}'.format(image[0], frame_count, frame[0]))

            # If the video is cached, use it. Else read frame from video file
            if cache:
                # logger.debug("using cache {}".format(frame[0]))
                frame1 = cache[image[0] - 1]
                frame2 = cache[image[0]]
            else:
                # logger.debug("direct access {}".format(frame[0]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, image[0] - 1)
                ret1, frame1 = cap.read()
                ret2, frame2 = cap.read()
                # print("Ret1 and 2: {},{}".format(ret1, ret2))

            frame_result = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

            # logger.debug("Writing to: {}".format(file_name))
            return_value = return_value and cv2.imwrite(file_name, frame_result)
        else:
            # logger.debug("Image already exists, skipping processing...")
            pass

    cap.release()

    return return_value


def invoke_ffmpeg(target_fps, music_file, frame_folder, destiny_file):
    """
        Prepare ffmpeg command and execute
        target_fps is the number of frames per second for the movie
        codec based on x264 and aac audio (mobile phone proof settings)
        """
    # First, get all files, compensate for windows ffmpeg (missing glob)
    file_list = [file for file in os.listdir(frame_folder) if file.endswith('.png')]
    file_list.sort()

    with open("{}/files.txt".format(frame_folder), "w") as text_file:
        for image_file in file_list:
            text_file.write("file '{}'\r\n".format(image_file))

    # OK... Perhaps target_fps will be obsolete now as parameter...
    amount_of_images = len(file_list)
    length_of_audio_file = MP3(music_file).info.length
    logger.debug("Calculated FPS: {}".format(target_fps))
    target_fps = int((100 * amount_of_images / length_of_audio_file)) / 100
    logger.debug("Actual FPS: {}".format(target_fps))

    command = []
    command.append('ffmpeg')

    # Convert images into video
    command.append("-y -r {} -f concat -safe 0 -i {}/files.txt".format(target_fps, frame_folder))

    # Add soundtrack
    command.append('-i {}'.format(music_file))

    # Set video codec
    command.append('-vcodec libx264 -profile:v high -preset slow')
    command.append('-pix_fmt yuv420p')
    command.append('-vprofile baseline -movflags +faststart')
    command.append('-strict -2 -acodec aac -b:a 128k')

    # Cut video/audio stream by the shortest one
    # command.append('-shortest')

    # Filename
    command.append('{}'.format(destiny_file))

    command = ' '.join(command)

    # print(command)
    result = os.system(str(command))
    logger.debug(result)


def main(folder_name, destiny_file, music_file, frame_folder, target_fps):
    logger.info("Starting main...")
    logger.info("Processing video folder: {}".format(folder_name))

    logger.debug("First, check the audio file...")
    # Determine the length of music file
    audio_file = MP3(music_file)
    logger.info("Length of audio file: {:0.1f} sec".format(audio_file.info.length))

    # Calculate the total amount of frames needed
    amount_of_frames_needed = int(target_fps * audio_file.info.length)
    logger.info("Amount of frames needed: {}".format(amount_of_frames_needed))

    # Load the image containing the figures to recognize.
    digits = load_reference_image('cijfers.png')

    # Load the list of video files to process
    raw_material = [avi_file for avi_file in glob.glob('{0}/*/*.AVI'.format(folder_name))]

    # Create a place where to put non-recognized/error time frames
    error_folder = "{}/error".format(folder_name)
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)

    # Recognize the timestamps in the video files
    logger.info("Analyzing timestamps in source videos...")
    with multiprocessing.Pool() as pool:
        partial_map = partial(determine_timestamps, digits=digits, error_folder=error_folder)
        timestamps = pool.map(partial_map, raw_material)

    # Calculate amount of frames available
    amount_of_frames_available = sum([i[1] for i in timestamps])
    logger.info("Amount of frames available: {}".format(amount_of_frames_available))

    # Reduce the amount of frames if needed
    start_time = False
    stop_time = False
    if amount_of_frames_available > amount_of_frames_needed:
        # There are more frames than needed, reduce the amount of frames
        timestamps, start_time, stop_time = select_timestamps(amount_of_frames_needed, timestamps)

    # Check again how much frames are available, better safe than sorry...
    amount_of_frames_available = sum([i[1] for i in timestamps])
    target_fps = int(100 * amount_of_frames_available / audio_file.info.length + 1) / 100
    logger.info('Calculated frame rate: {:0.2f}'.format(target_fps))

    # If needed create a folder for the processed image files
    logger.info("Frame folder: {}".format(frame_folder))
    if not os.path.exists(frame_folder):
        logger.debug("Image folder not existing, creating...")
        os.makedirs(frame_folder)

    # Clean up the image folder, remove unneeded files...
    logger.info("Cleaning up frame folder...")
    cleanup_image_folder(frame_folder, start_time, stop_time)

    # Process the selected frames
    logger.info("Processing selected frames...")
    with multiprocessing.Pool(processes=1) as pool:
        partial_map = partial(process_frames, destination_folder=frame_folder)
        result = pool.map(partial_map, timestamps)        # TODO: check result

    if all(result):
        logger.info("All frames processed OK")
    else:
        logger.error("Errors occurred during the processing of frames...")
    # for iets in timestamps:
    #    process_frames(iets, destination_folder=frame_folder)

    # Invoke ffmpeg
    invoke_ffmpeg(target_fps, music_file, frame_folder, destiny_file)

    logger.info('All done...')


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-f", "--frame-rate", help="Set target frame rate, default=30 fps", default=30)
    parser.add_argument("source_folder", help="Directory containing source video files")
    parser.add_argument("image_folder", help="Directory where the processed images are stored")
    parser.add_argument("audio_file", help="Audio file for time lapse video")
    parser.add_argument("destiny_file", help="Target video file")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # If we're started directly, call main() via a callable to measure performance
    t = timeit.Timer(lambda: main(
        args.source_folder,
        args.destiny_file,
        args.audio_file,
        args.image_folder,
        args.frame_rate))

    logger.info("Time needed to process: {:0.1f} sec".format(t.timeit(number=1)))