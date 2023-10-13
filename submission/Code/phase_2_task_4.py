import torch
import tensorly as tl
from tensorly.decomposition import parafac
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import csv
import numpy as np

def save_to_file(semantics, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for s in semantics:
            for label, weight in s:
                csv_writer.writerow([label, weight])
            csv_writer.writerow([])  # Empty row to separate different semantics

def connect_to_mongo():
    uri = "mongodb://localhost:27017/CSE515ProjectDB"
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client

def construct_three_modal_tensor(feature_space, labels, label_names):
    number_of_images = len(feature_space)
    feature_length = len(feature_space[0])
    num_labels = len(label_names)
    tensor = torch.randn((number_of_images, feature_length, num_labels), dtype=torch.float32)
    
    for i, (feature_vector, label) in enumerate(zip(feature_space, labels)):
        label_index = label_names.index(label)
        tensor[i, :, label_index] = torch.tensor(feature_vector, dtype=torch.float32)
    
    return tensor

def extract_latent_semantics(tensor_data, k, label_names):
    weights, factor_matrices = parafac(tensor=tensor_data, rank=k)
    label_factors = factor_matrices[2]  # Assuming the label mode is the third mode

    top_k_semantics = []
    for i in range(k):
        semantic_weights = label_factors[:, i]
        label_weight_pairs = list(zip(label_names, semantic_weights))
        sorted_pairs = sorted(label_weight_pairs, key=lambda x: x[1], reverse=True)
        top_k_semantics.append(sorted_pairs)

    return top_k_semantics

client = connect_to_mongo()
db = client.CSE515ProjectDB
collection = db.Phase2

tl.set_backend('pytorch')

print("Select one of the features: ")
print("1. Color Moment\n2. HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))

feature_names = ['color_moment', 'hog', 'layer3', 'avgpool', 'fc']
user_feature = feature_names[feature - 1]

feature_space = []
labels = []
labels_set = set()

for img in collection.find():
    feature_space.append(np.array(img[user_feature]).flatten())
    labels.append(img['target'])
    labels_set.add(img['target'])

label_names = sorted(list(labels_set))
tensor_data = construct_three_modal_tensor(feature_space, labels, label_names)

if torch.cuda.is_available():
    tensor_data = tensor_data.cuda()

latent_semantics = extract_latent_semantics(tensor_data, k, label_names)
file_name = f"latentsemantics_{k}_{user_feature}.csv"
save_to_file(latent_semantics, file_name)

print("Latent semantics saved to:", file_name)
