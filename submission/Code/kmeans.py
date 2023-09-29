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
rep_images = db.rep_images

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
    if (len(resized_img) != 3):
        continue
    if label == _label:
        resized_img = np.array([cv2.resize(i, (300, 100))
                               for i in img.numpy()])
        images.append(resized_img.flatten())
        image_ids.append(image_ID)
    else:
        kmeans = KMeans(n_clusters=num_clusters,
                        random_state=42).fit(images)
        cluster_centers = kmeans.cluster_centers_

        # Calculate distances of each image to the cluster centroids
        distances = cdist(images, cluster_centers, 'euclidean')
        closest_cluster_index = np.argmin(distances, axis=1)

        # Select an image closest to each cluster centroid
        representative_images[label] = [image_ids[i]
                                        for i in np.argmin(distances, axis=0)]
        print(label)
        label += 1
        images = []
        image_ids = []
        print(representative_images)
# Loop through each class
for label in range(101):
    images = features_coll.find({"target": label})
    images_data = np.array([(img["image"]).flatten() for img in images])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(images_data)
    cluster_centers = kmeans.cluster_centers_

    # Calculate distances of each image to the cluster centroids
    distances = cdist(images_data, cluster_centers, 'euclidean')
    closest_cluster_index = np.argmin(distances, axis=1)

    # Select an image closest to each cluster centroid
    representative_images[label] = [images[i]
                                    for i in np.argmin(distances, axis=0)]

# Display the representative images
plt.figure(figsize=(12, 12))
plt.suptitle('Representative Images for Each Class')

for i, (class_name, images) in enumerate(representative_images.items()):
    for j, img in enumerate(images):
        plt.subplot(len(representative_images),
                    num_clusters, i * num_clusters + j + 1)
        plt.imshow(img)
        plt.title(f'Class: {class_name}\nCluster: {j+1}')
        plt.axis('off')

plt.tight_layout()
plt.show()
