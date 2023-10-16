import numpy as np
from scipy.spatial import distance
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def connect_to_mongo():
    uri = "mongodb://localhost:27017/CSE515ProjectDB"
    client = MongoClient(uri, server_api=ServerApi('1'))
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)
    return client

client = connect_to_mongo()
db = client.CSE515ProjectDB
collection = db.Phase2

def get_feature_for_image(image_id, feature_name):
    """
    Fetch the desired feature for a specific image ID from the MongoDB collection.
    """
    image_data = collection.find_one({"_id": image_id})
    if image_data and feature_name in image_data:
        return np.array(image_data[feature_name])
    else:
        raise ValueError(f"Feature '{feature_name}' not found for image ID '{image_id}'.")

def get_all_features(feature_name):
    """
    Fetch the desired feature for all images from the MongoDB collection.
    """
    features = []
    for image_data in collection.find():
        if feature_name in image_data:
            features.append(np.array(image_data[feature_name]))
        else:
            features.append(None)  # or handle missing feature in another way
    return features

def project_to_latent_space(feature, latent_semantics):
    """
    Project a feature vector to a latent space defined by the provided latent semantics.
    """
    return np.dot(feature, latent_semantics)

def find_most_similar_images(query_feature, dataset_features, k):
    """
    Find the most similar images to the query feature based on Euclidean distance in the feature space.
    """
    distances = {}
    for idx, feature in enumerate(dataset_features):
        d = distance.euclidean(query_feature, feature)
        distances[d] = idx

    # Sort distances and get the top k indices
    sorted_distances = sorted(distances.keys())
    most_similar_indices = [distances[dist] for dist in sorted_distances[:k]]
    
    return most_similar_indices, sorted_distances[:k]

def find_most_similar_images_with_cosine(query_feature, dataset_features, k):
    """
    Find the most similar images to the query feature based on cosine similarity in the feature space.
    """
    similarities = {}
    for idx, feature in enumerate(dataset_features):
        sim = 1 - distance.cosine(query_feature, feature)  # cosine similarity = 1 - cosine distance
        similarities[sim] = idx

    # Sort similarities in descending order and get the top k indices
    sorted_similarities = sorted(similarities.keys(), reverse=True)
    most_similar_indices = [similarities[sim] for sim in sorted_similarities[:k]]
    
    return most_similar_indices, sorted_similarities[:k]

query_image = str(input("Enter the query image ID:"))
print("1. SVD.\n2. NNMF.\n3.LDA.\n4.K Means\n", end="")
ls = int(input("Select one of the above: "))

k_val = int(input("Enter how many output images are required:"))
ls_k = str(input("Enter the latent sematic dimensionality: "))
feature_names = ["color_moment", "hog", "layer3", "avgpool", "fc"]
print("1. Color Moment\n2. HoG.\n3. Layer 3.\n4. AvgPool.\n5. FC.")
feature = int(input("Select one of the feature space from above:"))

if ls == 1:
    # Project the query image feature and dataset features to the latent space
    query_image_features = get_feature_for_image(query_image, feature_names[feature - 1])
    
    # Loading the latent semantics
    path = f"C:/CSE 515 Assignment Pase 2/fantastic-umbrella/submission/Code/svd/svd_{ls_k}_latent_semantics_{feature_names[feature - 1]}.csv"
    latent_semantics = pd.read_csv(path, header=None).values
    
    # Getting the dataset features
    dataset_features = get_all_features(feature_names[feature - 1])

    query_feature_projected = project_to_latent_space(query_image_features, latent_semantics)
    dataset_features_projected = [project_to_latent_space(feature, latent_semantics) for feature in dataset_features]
    
    # Find the most similar images
    similar_image_indices, distances = find_most_similar_images(query_feature_projected, dataset_features_projected, 5)
    
    similar_image_indices, distances
