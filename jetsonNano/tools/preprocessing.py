import numpy as np
from skimage.transform import SimilarityTransform
import cv2
from mtcnn import MTCNN

import torch
from torch import nn 
from torch.nn import functional as F
from torchvision.transforms import InterpolationMode


def detectFaces(image): 
    """
    Using the MTCNN model all the faces in the input image are detected

    Args: 
        image: np.array()
    Outputs:
        faces: dict containing {'box', 'keypoints', 'confidence'} corresponding to each detected face
    """

    # Detect the faces using MTCNN 
    detector = MTCNN()
    faces = detector.detect_faces(image)

    return faces

def getFacialPoints(faces): 
    """"
    Extract the facial points correponding to the faces

    Args: 
        faces: dict containing {'box', 'keypoints', 'confidence'} corresponding to each detected face
    Outputs: 
        facialPtos: np.array() containing the facial landmarks
    """

    facialKeyPtos = []

    for i in range(len(faces)): 

        # Get all the keypoints
        keypoints = faces[i].get('keypoints')

        # Extract all 5 facial landmarks 
        left_eye = np.array(keypoints.get('left_eye')) 
        right_eye = np.array(keypoints.get('right_eye'))
        nose = np.array(keypoints.get('nose'))
        mouth_left = np.array(keypoints.get('mouth_left'))
        mouth_right = np.array(keypoints.get('mouth_right'))

        facialKeyPtos.append(np.array([left_eye, right_eye, nose, mouth_left, mouth_right], dtype=np.float32))

    return facialKeyPtos


def align_face(image: np.ndarray, src: np.array, dst: np.array = None, dsize = None): 
    """
    Alignment of a face given the source and target landmarks
    
    Args: 
        img: image of any size and any type
        src: landmarks of the source image 
        dst: landmarks for the target image
    Outputs: 
        image with size dsize and same type as img
    """

    if dst is None: 
        dst = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
        dtype=np.float32,) / 112


    if dsize is None: 
        dsize = (256,256)

    assert src.shape == (5,2), "Wrong shape of source landmarks"
    assert dst.shape == (5,2), "Wrong shape of destination landmarks"

    t_matrix = SimilarityTransform()
    t_matrix.estimate(src, dst*dsize)
    
    return cv2.warpAffine(image, t_matrix.params[0:2, :], dsize)


def extractFaces(image): 
    """
    
    Args: 

    Output:

    """

    faces = detectFaces(image)
    facialPtos = getFacialPoints(faces)

    # ... all the faces
    alignedFaces = []

    for i in range(len(faces)):
        alignedFaces.append(align_face(image, facialPtos[i]))

    return alignedFaces

