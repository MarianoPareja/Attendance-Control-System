from .models.inceptionResnetV1 import * 
import os
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

model = InceptionResnetV1(pretrained='vggface2')

features = []

feature_transforms = transforms.Compose([transforms.Resize((160, 160)),
                                         transforms.ToTensor()])

images_path = r'../data/facesDataset'

for image in os.listdir(images_path):
    img = Image.open(os.path.join(images_path, image))
    tens_image = feature_transforms(img).unsqueeze()
    
