import cv2
import numpy as np
import socket 
import threading
import time

from jetsonFunc import * 

# Initialize camera 
capture = cv2.VideoCapture(CAMERA_SOURCE)

threading.Thread(target=capture_image).start()      # Main Thread
