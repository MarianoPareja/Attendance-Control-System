import cv2
import os
import sys 
sys.path.append('..')

from tools.helpers import *
from tools.preprocessing import *

def createDataset():
    # Create dataset of align faces
    # file_path = os.path.abspath(__file__)
    # main_path = os.path.dirname(os.path.dirname(file_path))
    # images_path = os.path.join(main_path, 'data\classroomUCB_cellphone')

    images_path = r'C:\Users\Mariano\Desktop\test'

    cont = 0

    for img in os.listdir(images_path):

        # Read the image 
        image = cv2.imread(os.path.join(images_path, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect all the faces
        faces = detectFaces(image) 
        
        # Get facial points
        facialPtos = getFacialPoints(faces)
        
        # Align the faces
        for i in range(len(faces)): 
            face = align_face(image, facialPtos[i], dsize=(160,160))
            cv2.imwrite(r'C:\Users\Mariano\Desktop\test\mariano' + '%05d.jpg' %cont, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            # cv2.imwrite(os.path.join(os.path.join(main_path, 'data/facesDataset/student')) + '%05d.jpg' %cont, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            cont += 1
        
    return True

createDataset()