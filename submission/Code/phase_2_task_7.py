import numpy as np
import cv2
import torch
import pandas as pd
import torchvision
from torchvision.transforms import transforms
from scipy.spatial import distance
from sklearn.decomposition import PCA

from color_moment import extract_color_moment
from hog import extract_hog
from output_plotter import output_plotter
from resnet import extract_from_resnet
from database_connection import connect_to_mongo

query_image = str(input("Enter the query image ID:"))
print("1. SVD.\n2. NNMF.\n3.LDA.\n4.K Means\n", end="")
ls = int(input("Select one of the above: "))

k_val = int(input("Enter how many output images are required:"))
ls_k = str(input("Enter the latent sematic dimensionality: "))
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_ls1

# Get features of query image
transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)
img, label = dataset[int(query_image)]
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
query_image_feature = np.array(
    query_image_features[feature_names[feature - 1]]).flatten()
# calculating cov matrix for the input image feature
cov_matrix = np.outer(query_image_feature, query_image_feature)

# Calculating eigen vectors and eigen values
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Selecting the top ls_k eigen vectors
sorted_indices = np.argsort(eigenvalues)[::-1]
top_eigenvectors = eigenvectors[:, sorted_indices[:int(ls_k)]]

# project the higher dimensional feature into the latent space
query_image_feature_projected = np.dot(
    top_eigenvectors.T, query_image_feature)

match ls:
    case 1:
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "svd", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            d = distance.euclidean(
                query_image_feature_projected.flatten(), image_ls["latent_semantic"])
            distances[d] = image_ls["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])

    case 2:
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "nnmf", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            d = distance.euclidean(
                query_image_feature_projected.flatten(), image_ls["latent_semantic"])
            distances[d] = image_ls["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])
    case 3:
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "lda", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            d = distance.euclidean(
                query_image_feature_projected.flatten(), image_ls["latent_semantic"])
            distances[d] = image_ls["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])
    case 4:
        print("kmeans")
