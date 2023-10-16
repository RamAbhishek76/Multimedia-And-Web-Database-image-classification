# Task 10: Implement a program which, given (a) a label l, (b) a user selected latent semantics, and (c) positive integer k,
# identifies and lists k most relevant images, along with their scores, under the selected latent space.
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans
from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
features_coll = db.phase2_features
rep_images = db.phase2_representative_images
ls_collection = db.phase2_ls1
ls3_collection = db.phase2_ls3

input_label = int(input("Enter the input image label"))

# Latent semantic selection
dim_red_names = ["svd", "nnmf", "lda", "kmeans"]
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG\n3. Layer 3\n4. AvgPool\n5. FC\n6. Label-Label Similarity Matrix\n7. Image-Image Similarity Matrix")
feature = int(input("Select one feature from above: "))
print("1. SVD\n2. NNMF\n3. LDA\n4. K Means")
dim_red = int(
    input('Select one of the above dimensionality reduction methods: '))
ls_k = int(input("Enter the latent feature space dimensionality: "))

# Top k images
top_k = int(input("Enter k value for top k: "))

if feature != 6 and feature != 7:
    # Getting the rep image for the input label
    input_label_details = rep_images.find_one(
        {"target": int(input_label), "feature": feature_names[feature - 1]})
    input_label_feature = np.array(
        input_label_details["feature_value"]).flatten()

    distances = {}
    latent_space = []

    for image in ls_collection.find({"ls_k": int(ls_k), "dim_red_method": dim_red_names[dim_red - 1], "feature_space": feature_names[feature - 1]}):
        latent_space.append(np.array(image['latent_semantic']).flatten())

    kmeans = KMeans(n_clusters=len(input_label_feature), random_state=42)
    kmeans.fit(latent_space)

    # Get the cluster centroids (representative datapoints)
    representative_datapoints = kmeans.cluster_centers_

    # q, _ = np.linalg.qr(representative_datapoints)

    input_label_feature = input_label_feature @ representative_datapoints

    print("Calculating Latent Space distances")
    for image in ls_collection.find({"ls_k": int(ls_k), "dim_red_method": dim_red_names[dim_red - 1], "feature_space": feature_names[feature - 1]}):
        d = distance.euclidean(
            np.array(image['latent_semantic']).flatten(), input_label_feature)
        l = features_coll.find_one({"image_id": image["image_id"]})

        distances[d] = l["target"]
else:
    print("1. Color Moment\n2. HoG\n3. Layer 3\n4. AvgPool\n5. FC")
    feature = int(input("Select one feature from above: "))
    # Getting the rep image for the input label
    input_label_details = rep_images.find_one(
        {"target": int(input_label), "feature": feature_names[feature - 1]})
    input_label_feature = np.array(
        input_label_details["feature_value"]).flatten()

    distances = {}
    latent_space = []

    for image in ls_collection.find({"ls_k": int(ls_k), "dim_red_method": dim_red_names[dim_red - 1], "feature_space": feature_names[feature - 1]}):
        latent_space.append(np.array(image['latent_semantic']).flatten())

    kmeans = KMeans(n_clusters=len(input_label_feature), random_state=42)
    kmeans.fit(latent_space)

    # Get the cluster centroids (representative datapoints)
    representative_datapoints = kmeans.cluster_centers_

    # q, _ = np.linalg.qr(representative_datapoints)

    input_label_feature = input_label_feature @ representative_datapoints

    print("Calculating Latent Space distances")

    for image in ls3_collection.find({"ls_k": int(ls_k), "dim_red_method": dim_red_names[dim_red - 1], "feature_space": feature_names[feature - 1]}):
        print(image["image_id"])
        d = distance.euclidean(
            np.array(image['latent_semantic']).flatten(), input_label_feature)
        l = features_coll.find_one({"image_id": str(image["image_id"])})

        if not l:
            continue

        distances[d] = l["target"]

    dist_keys = sorted(list(distances.keys()))

dist_keys = sorted(list(distances.keys()))

print(f"Top {top_k} labels for query label {input_label}")
print("| Label | Score")
labels = []
for i in range(top_k):
    if distances[dist_keys[i]] not in (labels):
        labels.append(distances[dist_keys[i]])
        print(f'| {distances[dist_keys[i]]} | {max(dist_keys)/dist_keys[i]} |')
    if len(set(labels)) == top_k:
        break
