import os
import cv2
import torchvision
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
avg_coll = db.avg_images
features_coll = db.features
rep_images = db.representative_images

num_clusters = 10  # Number of clusters (you can adjust this)

# Dictionary to store representative images for each class
representative_images = {}

transforms = transforms.Compose([
    transforms.ToTensor(),
])

dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

label = 0
images = []
image_ids = []

for image_ID in range(8677):
    img, _label = dataset[image_ID]
    resized_img = np.array([cv2.resize(i, (300, 100))
                            for i in img.numpy()])
    if (len(resized_img) == 3):
        if label == _label:
            resized_img = np.array([cv2.resize(i, (300, 100))
                                    for i in img.numpy()])
            images.append(resized_img.flatten())
            image_ids.append(image_ID)
        else:
            if len(images):
                kmeans = KMeans(n_clusters=num_clusters,
                                random_state=42).fit(images)
                cluster_centers = kmeans.cluster_centers_

                # Calculate distances of each image to the cluster centroids
                distances = cdist(images, cluster_centers, 'euclidean')
                closest_cluster_index = np.argmin(distances, axis=1)

                # Select an image closest to each cluster centroid
                representative_images[label] = [image_ids[i]
                                                for i in np.argmin(distances, axis=0)]
                rep_images.insert_one(
                    {"target": label, "image_id": str(image_ids[np.argmin(distances, axis=0)[0]])})
            else:
                representative_images[label] = []
                rep_images.insert_one(
                    {"target": label, "image_id": ""})
            print(label)
            label += 1
            images = []
            image_ids = []
            print(representative_images)

for image_ID in range(8677):
    img, _label = dataset[image_ID]
    resized_img = np.array([cv2.resize(i, (300, 100))
                            for i in img.numpy()])
    if (len(resized_img) == 3):
        if label == _label:
            resized_img = np.array([cv2.resize(i, (300, 100))
                                    for i in img.numpy()])
            images.append(resized_img.flatten())
            image_ids.append(image_ID)
        else:
            if len(images):
                kmeans = KMeans(n_clusters=num_clusters,
                                random_state=42).fit(images)
                cluster_centers = kmeans.cluster_centers_

                # Calculate distances of each image to the cluster centroids
                distances = cdist(images, cluster_centers, 'euclidean')
                closest_cluster_index = np.argmin(distances, axis=1)

                # Select an image closest to each cluster centroid
                representative_images[label] = [image_ids[i]
                                                for i in np.argmin(distances, axis=0)]
                rep_images.insert_one(
                    {"target": label, "image_id": str(image_ids[np.argmin(distances, axis=0)[0]])})
            else:
                representative_images[label] = []
                rep_images.insert_one(
                    {"target": label, "image_id": ""})
            print(label)
            label += 1
            images = []
            image_ids = []
            print(representative_images)
