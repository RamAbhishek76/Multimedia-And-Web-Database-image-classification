import numpy as np
import cv2
import torch
import pandas as pd
import torchvision
from torchvision.transforms import transforms
from scipy.spatial import distance
from color_moment import extract_color_moment
from hog import extract_hog
from output_plotter import output_plotter
from resnet import extract_from_resnet

query_image = str(input("Enter the query image ID:"))
print("1. SVD.\n2. NNMF.\n3.LDA.\n4.K Means\n", end="")
ls = int(input("Select one of the above: "))

k_val = int(input("Enter how many output images are required:"))
ls_k = str(input("Enter the latent sematic dimensionality: "))
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))


# # Get features of query image
# transforms = transforms.Compose([
#     transforms.ToTensor(),
# ])

# # Loading the dataset
# dataset = torchvision.datasets.Caltech101(
#     'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
# data_loader = torch.utils.data.DataLoader(
#     dataset, batch_size=4, shuffle=True, num_workers=8)
# img, label = dataset[int(query_image)]
# # resizing the image into 300x10 for Color moment and HoG computation
# resized_img = [cv2.resize(i, (300, 100)) for i in img.numpy()]
# # resizing the image into 224x224 to provide as input to the resnet
# resized_resnet_img = [cv2.resize(i, (224, 224)) for i in img.numpy()]
# query_image_features = {}
# # checking if the image has 3 channels
# if len(resized_img) == 3:
#     color_moment = extract_color_moment(resized_img)
#     hog = extract_hog(resized_img)
#     resnet_features = extract_from_resnet(resized_resnet_img)
#     query_image_features = {
#         "image": img.numpy().tolist(),
#         "target": label,
#         "color_moment": color_moment,
#         "hog": hog,
#         "avgpool": resnet_features["avgpool"],
#         "layer3": resnet_features["layer3"],
#         "fc": resnet_features["fc"],
#     }

# U, S, VT = np.linalg.svd(
#     np.array(query_image_features[feature_names[feature - 1]]).flatten())
# # print(U)
# U_k = U[:, :int(ls_k)]
# # print(S[:int(ls_k)])
# S_k = np.diag(S[:int(ls_k)])

# latent_semantics = np.dot(U_k, S_k)
# latent_semantics = np.append([[i for i in range(len(latent_semantics[0]))]],
#                              latent_semantics, axis=0)

# print(latent_semantics)

distances = {}

match ls:
    case 1:
        path = "D:/ASU/Fall Semester 2023 - 24/CSE515 - Multimedia and Web Databases/phase 1/submission/Code/task3_output/svd/svd_" + \
            ls_k+"_latent_semantics_" + feature_names[feature - 1] + ".csv"
        ls_df = pd.read_csv(path)
        latent_features = []
        for index, row in ls_df.iterrows():
            latent_features.append([row[i] for i in range(1, len(row))])

        for i in range(len(latent_features)):
            d = distance.euclidean(
                latent_features[int(query_image)], latent_features[i])
            distances[d] = i

        dist_keys = sorted(distances.keys())
        for i in range(k_val):
            print(distances[dist_keys[i]])

    case 2:
        print("NNMF")
    case 3:
        path = "D:/ASU/Fall Semester 2023 - 24/CSE515 - Multimedia and Web Databases/phase 1/submission/Code/task3_output/lda/lda_" + \
            ls_k+"_latent_semantics_" + feature_names[feature - 1] + ".csv"
        ls_df = pd.read_csv(path)
        for index, row in ls_df.iterrows():
            print(row[0], row[1:ls_k])
    case 4:
        print("kmeans")
