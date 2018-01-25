# timelapse_ocr

At the moment I am taking timelapse footage from a build site. The used camera 
is a Brinno TimeLapse Pro. This camera produces motion-JPEG avi files 
with file names like TLCxxxxx.AVI. It is too "simple" for the end result to just put all the recorded AVI files in a row. The final assembly would be too long to be interesting too look at. So, as the build takes quite a long time, the goal of this project is to take all the produced AVI files and determine the date
and time of each movie frame. Then decide whether to include the frame in the final 
movie clip. From the selected images a "walking" average will be calculated, to
smooth out lightning differences. 

# used tools

For me this is the first time that I'm using python in Windows. As editor I am using PyCharm's Free Community editor with python 3.6.4, with imutils, opencv and numpy installed from within PyCharm. Also ffmpeg 3.4.1 is used for opencv and in the encoding of the final movie clip.
