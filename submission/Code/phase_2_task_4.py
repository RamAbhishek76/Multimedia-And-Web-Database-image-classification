import torch
import tensorly as tl
from tensorly.decomposition import parafac
from torchvision import datasets, transforms, models
from PIL import Image
import cv2
import numpy as np
from database_connection import connect_to_mongo


def extract_latent_semantics(tensor_data, k):
    factors = parafac(tensor=tensor_data, rank=k)
    # Extract label-weight pairs from the factors
    # This assumes the third mode corresponds to labels
    # This assumes the third mode corresponds to labels
    print("factors", len(factors))
    print("factors", len(factors[0]))
    print("factors", len(factors[0][0]))
    label_weights = factors[1]
    return label_weights


client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_features

# Set tensorly backend to PyTorch
tl.set_backend('pytorch')

print("select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))
feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']
feature_space = []
user_feature = feature_names[feature - 1]
for img in collection.find():
    print(img['image_id'])
    feature_space.append(np.array(img[user_feature]).flatten())

feature_space = np.array(feature_space)
feature_space = torch.from_numpy(feature_space)
# tensor_data = torch.stack([feature_space, labels.float()], dim=1)

latent_semantics = extract_latent_semantics(feature_space, k)
print(len(latent_semantics))
print(latent_semantics[1])
print(latent_semantics[0].numpy())


#  latent_semantics = np.append([[i for i in range(len(new_centroids[0]))]],
#                                      new_centroids, axis=0)

#         print(latent_semantics)

#         file_name = "nnmf_" + str(k) + "_latent_semantics_" + \
#             feature_names[feature - 1] + ".csv"
#         np.savetxt(file_name,
#                    latent_semantics, delimiter=',', fmt='%f')
#         df = pd.read_csv(file_name)
#         header = [i for i in range(len(df))]
#         df.to_csv(file_name, index=True)
