import time
import numpy as np
import cv2
from sklearn.cluster import KMeans
import torch
import pandas as pd
import torchvision
from torchvision.transforms import transforms
from scipy.spatial import distance
from sklearn.decomposition import PCA

from color_moment import extract_color_moment
from hog import extract_hog
from output_plotter import output_plotter, task_7_output_plotter
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
features = db.phase2_features

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
latent_space = []

query_image_feature = np.array(
    query_image_features[feature_names[feature - 1]]).flatten()

for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "svd", "feature_space": feature_names[feature - 1]}):
    latent_space.append(image_ls["latent_semantic"])

kmeans = KMeans(n_clusters=len(query_image_feature), random_state=42)
print("latent", latent_space)
kmeans.fit(latent_space)

# Get the cluster centroids (representative datapoints)
representative_datapoints = kmeans.cluster_centers_

q, _ = np.linalg.qr(representative_datapoints)

query_image_feature = query_image_feature @ q
print(len(representative_datapoints), len(representative_datapoints[0]))
print(len(query_image_feature))

match ls:
    case 1:
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "svd", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            d = distance.euclidean(query_image_feature,
                                   image_ls["latent_semantic"])
            print(d, image_ls["image_id"])
            distances[d] = image_ls["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])

        task_7_output_plotter(distances, dist_keys, k_val, query_image)

    case 2:
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "nnmf", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            d = distance.euclidean(query_image_feature,
                                   image_ls["latent_semantic"])
            distances[d] = image_ls["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])
    case 3:
        latent_semantics = []
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "lda", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            latent_semantics.append(image_ls["latent_semantic"])

        processed_latent_semantics = []

        for i in features.find():
            processed_latent_semantics.append(
                {"image_id": i["image_id"], "feature": [distance.euclidean(np.array(i[feature_names[feature - 1]]).flatten(), j) for j in latent_semantics]})

        for processed_feature in processed_latent_semantics:
            d = distance.euclidean(
                processed_feature["feature"], query_image_feature)
            distances[d] = processed_feature["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])
        print(len(processed_latent_semantics))
    case 4:
        latent_semantics = []
        for image_ls in collection.find({"ls_k": int(ls_k), "dim_red_method": "kmeans", "feature_space": feature_names[feature - 1]}):
            print(image_ls['image_id'], image_ls["feature_space"])
            latent_semantics.append(image_ls["latent_semantic"])

        processed_latent_semantics = []

        for i in features.find():
            processed_latent_semantics.append(
                {"image_id": i["image_id"], "feature": [distance.euclidean(np.array(i[feature_names[feature - 1]]).flatten(), j) for j in latent_semantics]})

        for processed_feature in processed_latent_semantics:
            d = distance.euclidean(
                processed_feature["feature"], query_image_feature)
            distances[d] = processed_feature["image_id"]

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]], dist_keys[i])
        print(len(processed_latent_semantics))
