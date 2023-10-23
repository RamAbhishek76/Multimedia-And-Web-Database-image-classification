# • Task 8: Implement a program which, given (a) an (even or odd numbered) imageID or an image file name, (b) a user
# selected latent semantics, and (c) positive integer k, identifies and lists k most likely matching labels, along with their
# scores, under the selected latent space.
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


def first_k_unique(lst, k):
    unique_values = set()
    result = []

    for value in lst:
        if value not in unique_values:
            unique_values.add(value)
            result.append(value)

            if len(result) == k:
                break

    return result


def cluster_distance_calculator(label, target, image_id, image_weight_pairs):
    sum_of_weights = 0
    count = 0
    selected_target = None

    # Iterate through all images
    for iterator in range(len(target)):
        # Check if the label matches the current image_id
        if int(label) == int(image_id[iterator]):
            selected_target = target[iterator]
            sum_of_weights += image_weight_pairs[iterator, 1]
            count += 1
    # Calculate average weight
    average_weight = sum_of_weights / count if count > 0 else 0

    return average_weight, selected_target


query_image_id = str(input("Enter the image id:"))
print("1. SVD.\n2. NNMF.\n3.LDA.\n4.K Means\n", end="")
ls = int(input("Select one of the above: "))

k_val = int(input("Enter how many output labels are required:"))
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

image_weight_pairs = np.loadtxt(
    f"imageweightpairs_task6_{ls}_{feature}.csv", delimiter=",")

weight_of_the_query_image = image_weight_pairs[int(query_image_id), 1]
# query_image_target = 1000

for i_target in range(len(image_id)):
    if int(image_id[i_target]) == int(query_image_id):
        query_image_target = target[i_target]

# Calculate absolute differences between weights
differences = image_weight_pairs[:, 1] - weight_of_the_query_image

# Get the indices of the k smallest differences
closest_indices = np.argsort(differences)

# Get the k closest image ids and weights
closest_image_ids = image_weight_pairs[closest_indices, 0]
closest_weights = image_weight_pairs[closest_indices, 1]
closest_image_ids_list = image_weight_pairs[closest_indices, 0].astype(
    int).tolist()
selected_target = []

for i in range(len(closest_image_ids_list)):
    index = 0
    selected_input_image_id = closest_image_ids_list[i]
    for iterator in range(len(image_id)):
        if int(image_id[iterator]) == selected_input_image_id:
            index = iterator
            break
    selected_target.append(target[int(index)])

unique_elements = first_k_unique(selected_target, k_val)

print("\nThe most relevant labels for the image id", query_image_id, "is:")

for i in range(len(unique_elements)):
    distance = cluster_distance_calculator(
        unique_elements[i], target, image_id, image_weight_pairs)
    print("Label:", unique_elements[i], "Similarity score:", cluster_distance_calculator(
        unique_elements[i], target, image_id, image_weight_pairs))
