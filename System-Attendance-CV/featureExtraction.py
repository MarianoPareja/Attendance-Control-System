from models.inceptionResnetV1 import *

import numpy as np 
from torchvision import transforms
import json
import datetime

def extractFeatVectors(images):
    """
    Extract a 128 long-vector describing the features of the parsed image
    Args: 
        model
        images(List): List of PIL.Images
    Output: 
        ftVector: np.array()
    """
    model = InceptionResnetV1(pretrained='vggface2').eval()

    transform = transforms.ToTensor()

    stacked_images = torch.stack([transform(img) for img in images])

    ftVectors = model.forward(stacked_images).detach()

    return ftVectors.numpy()

def cosineDistance(ftVectorA, ftVectorB):
    """
    Compute the cosine distance between two vectors
    Args: 
        ftVectorA: np.array()
        ftVectorB: np.array()
    Output: 
        distance{int}: 
    """
    if not (len(ftVectorA) == len(ftVectorB)):
        raise Exception("Both vector must have the same dimension")
    
    dot_product = np.dot(ftVectorA, ftVectorB)
    normA = np.linalg.norm(ftVectorA)
    normB = np.linalg.norm(ftVectorB)
    distance = 1 - dot_product / (normA * normB)

    return distance


def euclidianDistance(ftVectorA, ftVectorB):
    """
    Compute the euclidian distance between two vectors.
    Args: 
        ftVectorA: np.array()
        ftVectorB: np.array()
    Output: 
        distance: int
            0 -> More similarity
            Large -> Less similarity
    """
    if not (len(ftVectorA) == len(ftVectorB)):
        raise("Both vector must have the same dimension")
    
    distance = np.linalg.norm(ftVectorA - ftVectorB)

    return distance


def compareDatabase(ftVectors): 
    """
    Compare the similarity between an input image and pre-processed vectores in the database
    Args: 
        ftVectors: List containing np.array()
        mode: 1: Cosine Distance, 2: Euclidian Distance
    Ouputs:
        identity{List}: List containing the names of the students
    """
    THRESHOLD = 0.4

    if not len(ftVectors > 0): 
        return []
    
    # Path to the database 
    base_path = os.path.dirname(__file__)
    db_embeddeds_path = os.path.join(base_path, 'Database/embeddeds')

    # Initialize empty list
    identity = []
    
    for embedded in os.listdir(db_embeddeds_path): 
        file_path = os.path.join(db_embeddeds_path, embedded)
        with open(file_path, 'r') as json_file: 
            data = json.load(json_file)

            # Extract student name
            student_name = list(data.keys())[0]
            # Extract anchor embedded
            anchor_embeded = np.array(data[student_name])
            distances = [cosineDistance(anchor_embeded, ftVector) for ftVector in ftVectors]
        
        # Append the name to the list
        if min(distances) < THRESHOLD: 
            identity.append(student_name)
    
    return identity

            
def checkAssistance(studentList): 
    """
    
    Args:
        studentList{List}: List containing the name of the students ["Student 1", "Student 2", "Student 3"]
    Output: 
        -
    """
    actualTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try: 
        with open("register.txt", "w") as register: 
            register.write(actualTime + "\n")
            for name in studentList: 
                register.write(name + "\n")
    except IOError:
        print("Error al abrir el archivo")
    finally: 
        register.close()