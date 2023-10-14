# • Task 8: Implement a program which, given (a) an (even or odd numbered) imageID or an image file name, (b) a user
# selected latent semantics, and (c) positive integer k, identifies and lists k most likely matching labels, along with their
# scores, under the selected latent space.
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import torch
import torchvision
from torchvision.transforms import transforms
import cv2

from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet
# Enter the query image ID
query_image_id = str(input('Enter query image ID'))
print("1. SVD.\n2. NNMF.\n3.LDA.\n4.K Means\n", end="")
dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

# Select one of the latent spaces
ls = int(input("Select one of the above: "))
ls_k = str(input("Enter the latent sematic dimensionality: "))
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the features:"))
# Enter the number of top similar labels to be printed
k_val = int(input("Enter how many output images are required:"))

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_ls1
re_collection = db.phase2_representative_images


# Get features of query image
transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)
img, label = dataset[int(query_image_id)]
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

query_image_feature = np.array(
    query_image_features[feature_names[feature - 1]]).flatten()

processed_query_feature = []

for image in re_collection.find({"feature": feature_names[feature - 1]}):
    print(len(image["feature_value"]), len(query_image_feature))
    if len(image['feature_value']) > 0:
        d = distance.euclidean(np.array(image["feature_value"]).flatten(
        ), np.array(query_image_feature).flatten())
        print(d)
        processed_query_feature.append(d)
    else:
        print("yes")
        processed_query_feature.append(max(processed_query_feature))

# ls = np.argpartition(processed_query_feature, 10)
# print(ls)

file_name = dim_red_names[ls - 1] + "_" + str(ls_k) + "_label_label_similarity_" + \
    feature_names[feature - 1] + ".csv"
ls_df = pd.read_csv(file_name)

latent_space = [i[1:] for i in ls_df.values.tolist()]

q, _ = np.linalg.qr(latent_space)

processed_query_feature = np.dot(processed_query_feature, q)

distances = {}
m = max([abs(i) for i in processed_query_feature])
processed_query_feature = [i/m for i in processed_query_feature]

dis = []
for i in range(len(latent_space)):
    d = distance.cosine(processed_query_feature, latent_space[i])*100
    distances[d] = i
    dis.append(d)

# dist_keys = sorted(distances.keys())
# for i in range(k_val):
#     print(distances[dist_keys[i]], dist_keys[i])

ls = np.argpartition(dis, 10)
print(ls)
print(dis)
