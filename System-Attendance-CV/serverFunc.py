import socket 
import threading
from PIL.Image import frombytes
from featureExtraction import *
from datetime import datetime


def processData(data): 
    """
    Takes data in bytes and separates in batches to re-build the images
    Args: 
        data(List): List containing bytes of information
    Output:
        images(List): List of PIL.Images
    """

    # Define image size 
    image_shape = (160, 160, 3) 
    image_size = image_shape[0] * image_shape[1] * image_shape[2]
    images = []

    # Separate the data in images
    offset = 0

    print(f"Total Images: {len(data)/image_size}")

    while offset < len(data): 
        batch = data[offset : offset + image_size]

        # Re-build the image
        image = frombytes('RGB', (160,160), batch)

        # Save the image
        images.append(image)

        offset += image_size

    return images

def saveEmbeddeds(ftVectores): 
    """
    Save feature vector as .csv files 
    Args: 
        ftVectors(np.array): Feature vectores of 526 dimensions
    Output: 
        -
    """

    saveTime = datetime.now().strftime("%Y%m%d_%H%M%S")

    real_path = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(real_path, 'embeddeds')

    for i, vector in enumerate(ftVectores):
        # Generate a new name based on the time
        file_name = f"embedded_{saveTime}_{i+1}.csv" 
        # Save the file 
        file_path = os.path.join(folder_path, file_name)
        np.savetxt(file_path, vector, delimiter=',')

def checkAssistance(studentList): 
    """
    
    Args:
        studentList{List}: List containing the name of the students ["Student 1", "Student 2", "Student 3"]
    Output: 
        -
    """
    actualTime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not len(studentList) != 0: 
        return

    try: 
        with open("register.txt", "a") as register: 
            register.write(actualTime + "\n")
            for name in studentList: 
                register.write(name + "\n")
    except IOError:
        print("Error al abrir el archivo")
    finally: 
        register.close()

def manage_client(client_socket): 
    """
    Manages the connection with each client
    Args: 
        client_socket: connection to the client
    Output:
        -
    """

    data = client_socket.recv(4096)
    recv_data = b''

    while data: 
        recv_data += data
        data = client_socket.recv(4096)

    # Re-build images from the data
    images = processData(recv_data)

    # Extract faces' features vectors
    if not (len(images) != 0): 
        client_socket.close
        return
        
    # Extract feature vectores
    ftVectors = extractFeatVectors(images)

    # Save embbeded vectores 
    # saveEmbeddeds(ftVectors)

    # Compare with database
    studentsList = compareDatabase(ftVectors)

    # Mark assistance for students 
    checkAssistance(studentsList)
    
    # Close client's socket connection 
    client_socket.close()

    return
