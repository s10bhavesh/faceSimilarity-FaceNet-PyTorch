
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from config import *
# print(device)

# Define MTCNN module
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)

# Define Inception Resnet V1 module
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

