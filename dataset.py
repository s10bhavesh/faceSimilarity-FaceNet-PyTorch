
from torch.utils.data import DataLoader
from torchvision import datasets
from config import *
 
# print(workers)

def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('facenet-pytorch/data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
