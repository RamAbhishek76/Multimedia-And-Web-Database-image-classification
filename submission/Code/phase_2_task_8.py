import time
import numpy as np
import cv2
from scipy import datasets
from sklearn.cluster import KMeans
import torch
import pandas as pd
import torchvision
from torchvision.transforms import transforms
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from database_connection import connect_to_mongo

query_image_id = str(input("Enter the image id:"))
print("1. SVD.\n2. NNMF.\n3.LDA.\n4.K Means\n", end="")
ls = int(input("Select one of the above: "))

k_val = int(input("Enter how many output images are required:"))
print("1.LS1 \n2.LS2 \n3.LS3 \n4.LS4")
ls_k = str(input("Enter the dimnensionality: "))
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

client = connect_to_mongo()
db = client.CSE515ProjectDB
collection = db.Phase2

distances = {}
# Assuming latent_space is a list of feature vectors
target = []
image_id = []

# Collect features for images with the target label
for image_ls in collection.find():
    target.append(image_ls["target"])
    image_id.append(int(image_ls["image_id"]))

image_weight_pairs = np.loadtxt(f"imageweightpairs_task6_{ls}_{feature}.csv", delimiter=",")

weight_of_the_query_image = image_weight_pairs[int(query_image_id), 1]

# Calculate absolute differences between weights
differences = np.abs(image_weight_pairs[:, 1] - weight_of_the_query_image)

# Get the indices of the k smallest differences
closest_indices = np.argsort(differences)

# Get the k closest image ids and weights
closest_image_ids = image_weight_pairs[closest_indices, 0]
closest_weights = image_weight_pairs[closest_indices, 1]
closest_image_ids_list = image_weight_pairs[closest_indices, 0].astype(int).tolist()
selected_image_ids = []
selected_target = []

for i in range(len(closest_indices)):
    if differences[i] != 0:
        selected_image_ids.append(closest_image_ids_list[i])

for i in range(len(selected_image_ids)):
    index = 0
    selected_input_image_id = selected_image_ids[i]
    for iterator in range(len(image_id)):
        if int(image_id[iterator]) == selected_input_image_id:
            index = iterator
            break
    selected_target.append(target[int(index)])

print(selected_target)
unique_elements = np.unique(selected_target)[:k_val]
print(unique_elements)
print("\nThe most relevant labels for the image id", query_image_id, "is:")

for i in range(len(unique_elements)):
    print("Label:", unique_elements[i])