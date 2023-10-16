#  Task 2a: Implement a program which, given (a) a query imageID or image file, (b) a user selected feature space, and
# (c) positive integer k, identifies and lists k most likely matching labels, along with their scores, under the selected
# feature space.
import torch
import torchvision
import cv2
import numpy as np
from scipy.spatial import distance
from torchvision.transforms import transforms

from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet
from output_plotter import task_2_output_plotter
from input_from_file import input_from_file

np.set_printoptions(suppress=True)

client = connect_to_mongo()
db = client.cse515_project_phase1
features_collection = db.features
rep_collection = db.phase2_representative_images

transforms = transforms.Compose([
    transforms.ToTensor(),
])
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

input_image_data = []
inp_image_id = bool(
    str(input("Do you want to input Image Path? (y/N)")) == "y")
print(inp_image_id)
if inp_image_id:
    input_image_data = input_from_file()
else:
    query_image_id = int(input("Enter the image ID:"))
    input_image_data = dataset[query_image_id][0]
print("select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))

feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']

# resizing the image into 300x10 for Color moment and HoG computation
resized_img = [cv2.resize(i, (300, 100)) for i in input_image_data.numpy()]
# resizing the image into 224x224 to provide as input to the resnet
resized_resnet_img = [cv2.resize(i, (224, 224))
                      for i in input_image_data.numpy()]

query_image_feature = []

match feature:
    case 1:
        color_moment = extract_color_moment(resized_img)
        query_image_feature = color_moment
    case 2:
        hog = extract_hog(resized_img)
        query_image_feature = hog
    case 3 | 4 | 5:
        resnet_features = extract_from_resnet(resized_resnet_img)
        match feature:
            case 3:
                query_image_feature = resnet_features["layer3"]
            case 4:
                query_image_feature = resnet_features["avgpool"]
            case 5:
                query_image_feature = resnet_features["fc"]

feature_ranks = {}
iids = []

for image in rep_collection.find({"feature": feature_names[feature - 1]}):
    if image:
        print(image["image_id"])
        image_feature = np.array(
            image["feature_value"]).flatten()
        if len(image_feature) > 0:
            match feature:
                case 1:
                    d_cm = distance.euclidean(
                        image_feature, np.array(query_image_feature).flatten())
                case 2:
                    d_cm = distance.cosine(
                        image_feature, np.array(query_image_feature).flatten())
                    print(image["image_id"], d_cm)
                case 3:
                    d_cm = distance.cosine(
                        image_feature, np.array(query_image_feature).flatten())
                case 4:
                    # cosine is giving acceptable results (the expected result is withing top 5)
                    d_cm = distance.cosine(
                        image_feature, np.array(query_image_feature).flatten())
                case 5:
                    d_cm = distance.cosine(
                        image_feature, np.array(query_image_feature).flatten())
            feature_ranks[d_cm] = image["image_id"]
c = 0
ids = []
cm_keys = sorted(feature_ranks.keys())

for i in cm_keys:
    print(c, len(set(ids)), set(ids))
    c += 1
    image_details = features_collection.find_one(
        {"image_id": feature_ranks[i]})

    ids.append(image_details["target"])
    if (len(set(ids)) == k + 1):
        break
    feature_ranks[i] = image_details["target"]

used_targets = []
print("| Label | Score |")
i = 0

while i < k:
    if feature_ranks[cm_keys[i]] not in used_targets:
        print("| " + str(feature_ranks[cm_keys[i]]) +
              " | " + str((max(cm_keys)/cm_keys[i]) - 1) + " |")
        i += 1
