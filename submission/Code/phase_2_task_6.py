from scipy.spatial import distance
import numpy as np

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
avg_coll = db.avg_images
features_coll = db.features
rep_images = db.new_representative_images

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

for i in range(8677):
    for j in range(i, 8677):
        print(i, j)
        image_features = list(features_coll.find(
            {"image_id": {"$in": [str(i), str(j)]}}))
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

for i in sim_matrix:
    for j in i:
        print(j, " ", end="")
    print()

list_sims = sorted(list(sim_matrix.flatten()))

ctr = 1
while k != ctr:
    ctr += 1
    print(list_sims[ctr - 1])
