from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import numpy as np
import networkx as nx
from scipy.spatial import distance

# Database Connection
def connect_to_mongo():
    uri = "mongodb://localhost:27017/CSE515ProjectDB"
    client = MongoClient(uri, server_api=ServerApi('1'))
    return client

client = connect_to_mongo()
db = client.CSE515ProjectDB
collection = db.Phase2

def get_feature_for_image(image_id, feature_name):
    image_data = collection.find_one({"image_id": image_id})
    if image_data and feature_name in image_data:
        return np.array(image_data[feature_name])
    else:
        return None

def get_all_features(feature_name):
    return [get_feature_for_image(str(i), feature_name) for i in range(1, 1001)]  # Assuming 1000 images

def construct_similarity_graph(features, n):
    num_images = len(features)
    adjacency_matrix = np.zeros((num_images, num_images))
    
    for i in range(num_images):
        distances = [distance.euclidean(features[i], f) if f is not None else float('inf') for f in features]
        k_nearest = np.argsort(distances)[:n]
        for j in k_nearest:
            adjacency_matrix[i][j] = 1
            
    return adjacency_matrix

def personalized_page_rank(adjacency_matrix, label_l, alpha=0.85, max_iter=100, tol=1e-6):
    num_nodes = adjacency_matrix.shape[0]
    outbound_strength = np.sum(adjacency_matrix, axis=1)
    
    transition_matrix = adjacency_matrix / outbound_strength[:, None]
    transition_matrix = np.nan_to_num(transition_matrix)
    
    p = np.zeros(num_nodes)
    p[label_l] = 1
    
    r = np.full(num_nodes, 1/num_nodes)
    for _ in range(max_iter):
        r_new = alpha * np.dot(transition_matrix, r) + (1 - alpha) * p
        if np.linalg.norm(r_new - r, 2) < tol:
            break
        r = r_new
    
    return r

def main():
    feature_name = input("Enter the feature model or latent space (color_moment, hog, layer3, avgpool, fc): ")
    n = int(input("Enter the value of n (number of similar images for graph): "))
    m = int(input("Enter the value of m (number of significant images to return): "))
    label_l = int(input("Enter the label l: "))

    features = get_all_features(feature_name)
    similarity_graph = construct_similarity_graph(features, n)
    significance_scores = personalized_page_rank(similarity_graph, label_l)
    
    top_m_indices = np.argsort(significance_scores)[-m:]
    print(f"Top {m} significant images (indices): {top_m_indices}")

main()
