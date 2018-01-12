# timelapse_ocr

At the moment I am taking timelapse footage from a build site. The used camera 
is a Brinno TimeLapse Pro camera. This camera produces motion-JPEG avi files 
with file names like TLC000xx.AVI. As the build takes quite a long time, the 
goal of this project is to take all the produced AVI files, determine the date
and time of each frame. Then decide whether to include the frame in the final 
movie clip. From the selected images a "walking" average will be calculated, to
smooth out lightning differences. 

# used tools

For me this is the first time that I'm using python in Windows. Used versions:
- Python 3.6.4 
- ffmpeg 3.4.1 
- numpy-1.14.0+mkl-cp36-cp36m-win32.whl
- opencv_python-3.4.0-cp36-cp36m-win32.whl

To be evaluated: amd64 versions. Will also need a different version of python,
I presume...
