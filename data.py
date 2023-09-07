import torchvision, torch
from torchvision import datasets, models, transforms
from pathlib import Path
import os

print("Torchvision version ", torchvision.__version__)

from torchvision.io import read_image, ImageReadMode
Path.cwd()
print(os.getcwd())
os.chdir('/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101')
Path.cwd()
print(os.getcwd())

dataset = torchvision.datasets.Caltech101('/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101')
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

print(dataset)
