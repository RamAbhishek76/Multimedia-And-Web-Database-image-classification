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
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

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
ls_k = str(input("Enter the latent semantic:"))
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))
k = str(input("Enter k value:"))

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_features

match(int(ls_k)):
    case 4:
        distances = {}
        # Assuming latent_space is a list of feature vectors
        target = []
        image_id = []

        # Collect features for images with the target label
        for image_ls in collection.find():
            target.append(image_ls["target"])
            image_id.append(int(image_ls["image_id"]))

        image_weight_pairs = np.loadtxt(f"imageweightpairs_task6_{ls}_{feature_names[feature-1]}_{k}.csv", delimiter=",")

        #weight_of_the_query_image = image_weight_pairs[int(query_image_id), 1]

        matching_rows = np.where(image_weight_pairs[:, 0] == int(query_image_id))[0]

        if matching_rows.size < 1:
            print("Image ID does not exist")
            exit

        # Get the first matching row (you can adjust this depending on your needs)
        first_matching_row = image_weight_pairs[matching_rows[0]]

        # Get the corresponding value from the second column
        weight_of_the_query_image = first_matching_row[1]

        # Calculate absolute differences between weights
        differences = abs(image_weight_pairs[:, 1] - weight_of_the_query_image)

        sum_of_weight_differences_of_all_targets = np.zeros(101)
        count_of_images_for_the_target = np.zeros(101)
        weights_of_the_labels = np.zeros(101)

        for images_iterator in range(len(image_id)):
            target_of_selected_image = target[images_iterator]
            difference_of_selected_image = differences[images_iterator]
            sum_of_weight_differences_of_all_targets[target_of_selected_image] += difference_of_selected_image
            count_of_images_for_the_target[target_of_selected_image] += 1

        for weight_iterator in range(101):
            weights_of_the_labels[weight_iterator] = sum_of_weight_differences_of_all_targets[weight_iterator] / count_of_images_for_the_target[weight_iterator]

        sorted_weights = np.argsort(weights_of_the_labels)

        print("\nThe most relevant labels for the image id", query_image_id, "is:")
        counter = 0
        for sorted_weight_iterator in sorted_weights[:k_val]:
            print("Label:", sorted_weight_iterator , "Similarity score:", weights_of_the_labels[counter])
            counter += 1
    case 3:
        collection = db.phase2_ls1
        re_collection = db.phase2_representative_images

        dim_red_names = ["svd", "nnmf", "lda", "kmeans"]
        feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]

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

        # Find distances of the query image to all the cluster centers
        processed_query_feature = []

        for image in re_collection.find({"feature": feature_names[feature - 1]}):
            if len(image['feature_value']) > 0:
                d = distance.euclidean(np.array(image["feature_value"]).flatten(
                ), np.array(query_image_feature).flatten())
                print(d)
                processed_query_feature.append(d)
            else:
                print("yes")
                processed_query_feature.append(max(processed_query_feature))

        print(np.argmin(processed_query_feature))

        file_name = dim_red_names[ls - 1] + "_" + str(k) + "_label_label_similarity_" + \
            feature_names[feature - 1] + ".csv"
        ls_df = pd.read_csv(file_name)

        latent_space = [i[1:] for i in ls_df.values.tolist()]

        latent_space_feature = latent_space[np.argmin(processed_query_feature)]

        distances = [distance.euclidean(i, latent_space_feature) for i in latent_space]

        res = np.argsort(distances)[:k_val]

        print(f'Top {k_val} labels for image {query_image_id}')
        for i in range(len(res)):
            print(f'{i + 1}. {res[i]}')