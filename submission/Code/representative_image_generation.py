import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from database_connection import connect_to_mongo


def gen_empty_feat_dict():
    image_features = {}
    image_ids = {}

    feature_names = ['color_moment',
                     'hog', 'avgpool', 'layer3', 'fc']

    for feature in feature_names:
        image_features[feature] = []
        image_ids[feature] = []

    return image_ids, image_features


client = connect_to_mongo()
db = client.cse515_project_phase1
avg_coll = db.avg_images
features_coll = db.features
rep_images = db.final_representative_images


def cluster_by_feature(feature_name):
    print("------------------------------clustering using ",
          feature_name, "------------------------------")
    for label in range(101):
        image_ids, image_features = [], []
        for image in features_coll.find({"target": label}):
            image_ids.append(image["image_id"])
            image_features.append(np.array(image[feature_name]).flatten())
            print(label, image["image_id"])
        print("Loaded images for label ", label)
        if len(image_features):
            kmeans = KMeans(n_clusters=1,
                            random_state=42).fit(image_features)
            cluster_centers = kmeans.cluster_centers_

            # Calculate distances of each image to the cluster centroids
            distances = cdist(
                image_features, cluster_centers, 'euclidean')
            closest_cluster_index = np.argmin(distances, axis=0)
            print(label, closest_cluster_index,
                  image_ids[int(closest_cluster_index)])
            rep_images.insert_one({"target": label, "image_id": str(
                image_ids[int(closest_cluster_index)]), "feature": feature_name, "feature_value": list(image_features[int(closest_cluster_index)])})
        else:
            rep_images.insert_one(
                {"target": label, "image_id": "", "feature": feature_name, "feature_value": []})


cluster_by_feature("color_moment")
cluster_by_feature("hog")
cluster_by_feature("layer3")
cluster_by_feature("avgpool")
cluster_by_feature("fc")
