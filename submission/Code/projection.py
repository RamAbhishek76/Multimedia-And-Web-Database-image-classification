import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision
from torchvision.transforms import transforms
from sklearn.cluster import KMeans

from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_ls1

latent_space = []
for image_ls in collection.find({"ls_k": int(5), "dim_red_method": "svd", "feature_space": 'layer3'}):
    latent_space.append(image_ls["latent_semantic"])

# latent_space_array = np.array(latent_space)

# U, S, VT = np.linalg.svd(latent_space_array)

# p_values = np.sqrt(S)

# Get features of query image
transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)
img, label = dataset[2500]
# resizing the image into 300x10 for Color moment and HoG computation
resized_img = [cv2.resize(i, (300, 100)) for i in img.numpy()]
# resizing the image into 224x224 to provide as input to the resnet
resized_resnet_img = [cv2.resize(i, (224, 224)) for i in img.numpy()]
query_image_features = {}
# checking if the image has 3 channels
if len(resized_img) == 3:
    color_moment = extract_color_moment(resized_img)
    hog = extract_hog(resized_img)
    resnet_features = extract_from_resnet(resized_resnet_img)
    query_image_features = {
        "image": img.numpy().tolist(),
        "target": label,
        "color_moment": color_moment,
        "hog": hog,
        "avgpool": resnet_features["avgpool"],
        "layer3": resnet_features["layer3"],
        "fc": resnet_features["fc"],
    }

distances = {}

K = 1024

# Apply K-means clustering
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(latent_space)

# Get the cluster centroids (representative datapoints)
representative_datapoints = kmeans.cluster_centers_

# Verify the shape of the representative datapoints
print("Shape of representative datapoints:", representative_datapoints.shape)
for i in representative_datapoints:
    for j in i:
        print(j, end=" ")
    print()
