# Task 10: Implement a program which, given (a) a label l, (b) a user selected latent semantics, and (c) positive integer k,
# identifies and lists k most relevant images, along with their scores, under the selected latent space.
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans
from database_connection import connect_to_mongo
from output_plotter import task_10_output_plotter

client = connect_to_mongo()
db = client.cse515_project_phase1
features_coll = db.phase2_features
rep_images = db.phase2_representative_images
ls_collection = db.phase2_ls1

input_label = int(input("Enter the input image label"))

# Latent semantic selection
dim_red_names = ["svd", "nnmf", "lda", "kmeans"]
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG\n3. Layer 3\n4. AvgPool\n5. FC")
feature = int(input("Select one feature from above: "))
print("1. SVD\n2. NNMF\n3. LDA\n4. K Means")
dim_red = int(
    input('Select one of the above dimensionality reduction methods: '))
ls_k = int(input("Enter the latent feature space dimensionality: "))

# Top k images
top_k = int(input("Enter k value for top k: "))

# Getting the rep image for the input label
input_label_details = rep_images.find_one(
    {"target": int(input_label), "feature": feature_names[feature - 1]})
input_label_feature = np.array(input_label_details["feature_value"]).flatten()

distances = {}
latent_space = []

for image in ls_collection.find({"ls_k": int(ls_k), "dim_red_method": dim_red_names[dim_red - 1], "feature_space": feature_names[feature - 1]}):
    latent_space.append(np.array(image['latent_semantic']).flatten())

kmeans = KMeans(n_clusters=len(input_label_feature), random_state=42)
kmeans.fit(latent_space)

# Get the cluster centroids (representative datapoints)
representative_datapoints = kmeans.cluster_centers_

q, _ = np.linalg.qr(representative_datapoints)

input_label_feature = input_label_feature @ q

print("Calculating Latent Space distances")

for image in ls_collection.find({"ls_k": int(ls_k), "dim_red_method": dim_red_names[dim_red - 1], "feature_space": feature_names[feature - 1]}):
    d = distance.euclidean(
        np.array(image['latent_semantic']).flatten(), input_label_feature)
    distances[d] = image["image_id"]

dist_keys = sorted(list(distances.keys()))

res = []
for i in range(top_k):
    print(distances[dist_keys[i]], dist_keys[i])
    res.append((distances[dist_keys[i]], dist_keys[i]))

task_10_output_plotter(res, input_label)
