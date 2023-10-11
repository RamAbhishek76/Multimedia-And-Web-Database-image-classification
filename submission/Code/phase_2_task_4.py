import torch
import tensorly as tl
from tensorly.decomposition import parafac
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np

# Set tensorly backend to PyTorch
tl.set_backend('pytorch')
transform = transforms.Compose([
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(
    root='D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=len(dataset), shuffle=False)

print("select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))

data, labels = next(iter(dataloader))

tensor_data = torch.stack([features, labels.float()], dim=1)


def extract_latent_semantics(tensor_data, k):
    factors = parafac(tensor_data, rank=k)
    # Extract label-weight pairs from the factors
    # This assumes the third mode corresponds to labels
    label_weights = factors[1]
    return label_weights


k = 5  # or any other value you choose
latent_semantics = extract_latent_semantics(tensor_data, k)
print(latent_semantics)
