import csv
import os
import pandas as pd
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD

from database_connection import connect_to_mongo

def save_to_file(matrix, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for row in matrix:
            csv_writer.writerow(row)

client = connect_to_mongo()
db = client.CSE515ProjectDB
features_coll = db.Phase2
avg_coll = db.avg_images
rep_images = db.phase2_representative_images

sim_matrix = np.zeros(shape=(8677, 8677))

print("Select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
print("Select one of the dimensionality reduction methods: ")
print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
dim_red_method = int(input("Choose one from above: "))

dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

k = int(input("Enter k value: "))

feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']

csv_file_name = f"latentsemantics_task6_{dim_red_method}_{feature}.csv"
text_file_name = f"latentsemantics_task6_{dim_red_method}_{feature}.txt"

if (not os.path.isfile("latentsemantics_task6_1_5.csv")):
    for i in range(0, 8677, 2):
        print(i)
        for j in range(i, 8677, 2):
            #print(i, j)
            image_features = []
            match feature:
                case 1:
                    image_features = list(features_coll.find(
                        {"image_id": {"$in": [str(i), str(j)]}}, {
                            "color_moment": 1
                        }))
                case 2:
                    image_features = list(features_coll.find(
                        {"image_id": {"$in": [str(i), str(j)]}}, {
                            "hog": 1
                        }))
                case 3:
                    image_features = list(features_coll.find(
                        {"image_id": {"$in": [str(i), str(j)]}}, {
                            "layer3": 1
                        }))
                case 4:
                    image_features = list(features_coll.find(
                        {"image_id": {"$in": [str(i), str(j)]}}, {
                            "avgpool": 1
                        }))
                case 5:
                    image_features = list(features_coll.find(
                        {"image_id": {"$in": [str(i), str(j)]}}, {
                            "fc": 1
                        }))
            if len(image_features) >= 2:
                sim = distance.euclidean(
                    np.array(image_features[0][
                        feature_names[feature - 1]]).flatten().reshape(1, -1).flatten(), np.array(image_features[1][
                            feature_names[feature - 1]]).flatten().reshape(1, -1).flatten())
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
            else:
                sim_matrix[i][j] = 0
                sim_matrix[j][i] = 0

    np.savetxt(text_file_name, sim_matrix, fmt='%.6f', delimiter='\t')
    print(f"Similarity matrix saved to {text_file_name}")

    save_to_file(sim_matrix, csv_file_name)
    print(f"Similarity matrix saved to {csv_file_name}")

similarity_matrix = np.loadtxt("latentsemantics_task6_1_3.csv", delimiter=",")

match dim_red_method:
    case 1:
    # SVD
        svd = TruncatedSVD(n_components=k, random_state=42)
        latent_semantics = svd.fit_transform(similarity_matrix)

    case 2:
    # NNMF
        nmf = NMF(n_components=k, init='random', random_state=42)
        latent_semantics = nmf.fit_transform(similarity_matrix)

    case 3:
    # LDA
        lda = LatentDirichletAllocation(n_components=k, random_state=42)
        latent_semantics = lda.fit_transform(similarity_matrix)

    case 4:
        # k-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_assignments = kmeans.fit_predict(similarity_matrix)
        # Iterate through each cluster
        cluster_centers = kmeans.cluster_centers_
        latent_semantics = np.dot(similarity_matrix, cluster_centers.T)
        print(latent_semantics.shape)

# Calculate image weights based on the first latent semantic (as an example)
image_weights = latent_semantics[:, 0]

# Create a list of image-weight pairs
image_weight_pairs = list(enumerate(image_weights))

# Sort by weights in decreasing order
sorted_image_weight_pairs = sorted(image_weight_pairs, key=lambda x: x[1], reverse=True)

# Display the top k image-weight pairs
for i in range(k):
    image_id, weight = sorted_image_weight_pairs[i]
    print(f"Image {image_id}: Weight {weight}")

csv_file_name_for_sortedweightpairs = f"imageweightpairs_task6_{dim_red_method}_{feature}.csv"
save_to_file(sorted_image_weight_pairs, csv_file_name_for_sortedweightpairs)
print(f"Sorted image weight pairs saved to {csv_file_name_for_sortedweightpairs}")

# # List image-weight pairs ordered by decreasing weights
# query_image_index = 0  # Replace this with the index of your query image
# query_image_vector = similarity_matrix[query_image_index]

# distances = {}
# for i, image_vector in enumerate(similarity_matrix):
#     if i != query_image_index:
#         d = distance.euclidean(query_image_vector, image_vector)
#         distances[d] = i

# # Sort distances and get the top k indices
# sorted_distances = sorted(distances.items(), key=lambda x: x[0])
# top_k_indices = [index for _, index in sorted_distances[:k]]

# # Print the image-weight pairs
# for index in top_k_indices:
#     print(f"Image {index}, Weight: {distances[sorted_distances[index][0]]}")