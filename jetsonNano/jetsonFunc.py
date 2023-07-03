import cv2
import time
import socket
import threading
from torchvision import transforms

from preprocessData import * 

FRAME_INTERVAL = 1000
CAMERA_SOURCE = 0

# Initialize camera 
capture = cv2.VideoCapture(CAMERA_SOURCE)

# Configure
HOST = 'localhost'
PORT = 8888

# Create client socket 
# IPv4 / TCP
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# def extract_faces():

def send_information(faces):
    """
    Sent all the faces to the server as bytes
    Args:
        faces{List}:  
    Output:
    
    """   
    # Pre-process data befores sending 
    faces_bytes = [face.tobytes() for face in faces]
    send_data = b"".join(faces_bytes)

    # Create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    while True: 
        try:
            # Stablish connection with the server 
            client_socket.connect((HOST,PORT))
            print("Connection stablished")

            client_socket.sendall(send_data)
            client_socket.close()
            break

        except socket.error as e: 
            print(f"Error connecting to the server {e}")
            time.sleep(5)       # Wait 5 seconds before trying again
    
    print("Data send succesfully")


def manage_image(image): 
    
    # Pre-process the image 
    preImg = MTCNN_preprocessing(image)

    # Extract faces
    faces = detectFaces(preImg)
    facialPtos = getFacialPtos(faces)
    alignFaces = alignAllFaces(image, facialPtos, None, (160,160))

    # Send faces to the server
    thread_sendIng = threading.Thread(target=send_information, args=(alignFaces,))
    thread_sendIng.start()
    thread_sendIng.join()
    

def capture_image(): 
    
    # Create VideoCapture instance
    capture = cv2.VideoCapture(CAMERA_SOURCE)
    capture_time = time.time()

    while True: 
        
        if (time.time() >= capture_time + (FRAME_INTERVAL / 60)): 
            
            # Update capture time 
            capture_time = time.time()

            # Make sure an image is read
            ret = False
            while not ret: 
                ret, image = capture.read()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Send image to a new thread
            threading.Thread(target=manage_image, args=(image,)).start()

            time.sleep(FRAME_INTERVAL/60)


