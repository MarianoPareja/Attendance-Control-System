import numpy as np
import matplotlib.pyplot as plt

def visualize_faces(alignedFaces):

    plt.figure(figsize=[15,10])
    
    rows = int(np.ceil(len(alignedFaces)/3))
    plt.subplot(rows, 3, 1)
    
    for i in range(len(alignedFaces)):
        plt.subplot(rows, 3, i+1), plt.imshow(alignedFaces[i])

    plt.show()