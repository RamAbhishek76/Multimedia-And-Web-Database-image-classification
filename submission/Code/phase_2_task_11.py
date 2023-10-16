import networkx as nx
import numpy as np
from database_connection import connect_to_mongo
from output_plotter import output_plotter_task_11
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.Caltech101(root='./data', download=True, transform=transform)

def fetch_data_from_db(descriptor, is_latent_space):
    """
    Fetch the desired feature for all images from the MongoDB collection.
    """
    client = connect_to_mongo()
    db = client.cse515
    if not is_latent_space:
        collection = db.Phase2
        data = []
        for image_data in collection.find():
            if descriptor in image_data:
                entry = {
                    'image_id': image_data['image_id'],
                    'feature': np.array(image_data[descriptor]),
                    'label': image_data['target']
                }
                data.append(entry)
    else:
        collection = db.merged_collection
        data = []
        k, feature, method = descriptor
        for image_data in collection.find():
            if k == image_data['ls_k'] and feature == image_data['feature_space'] and method == image_data['dim_red_method']:
                entry = {
                    'image_id': image_data['image_id'],
                    'feature': np.array(image_data['latent_semantic']),
                    'label': image_data['target']
                }
                data.append(entry)
    return data

def create_similarity_graph(data, n):
    """
    Create a similarity graph based on feature vectors.
    """
    G = nx.Graph()
    
    for i, entry in enumerate(data):
        G.add_node(entry['image_id'])
        
        # Compute similarities
        similarities = [np.dot(entry['feature'], other_entry['feature']) for other_entry in data]
        
        # Get top n most similar images
        top_n_indices = sorted(range(len(similarities)), key=lambda k: similarities[k])[-n:]
        
        for index in top_n_indices:
            G.add_edge(entry['image_id'], data[index]['image_id'])
            
    return G

def personalized_page_rank(G, label, m, data):
    """
    Compute personalized PageRank and get top m images.
    """
    personalization = {entry['image_id']: 1 if entry['label'] == label else 0 for entry in data}
    ranks = nx.pagerank(G, personalization=personalization)
    
    # Sort nodes by rank and get top m
    top_m_nodes = sorted(ranks, key=ranks.get, reverse=True)[:m]
    
    return top_m_nodes

def main(descriptor, n, m, label, is_feature_model):
    data = fetch_data_from_db(descriptor, is_feature_model)
    G = create_similarity_graph(data, n)
    significant_images = personalized_page_rank(G, label, m, data)
    output_plotter_task_11(dataset=dataset, feature_descriptor=descriptor, label=label, input_image_id_list=list(significant_images))
    return significant_images

feature_or_ls = input("Do you want to enter a feature Model? (Y/N): ")
is_latent_space = True if feature_or_ls.lower() == 'n' else False
if is_latent_space:
    print("Enter the latent space model: ")
    print("1. LS1 (T3)\n2. LS2 (T4)\n3. LS3 (T5)\n4. LS4 (T6)\n")
    ls = int(input("Choose one of the latent space model from above: "))
    if ls == 1:
        print("select one of the features: ")
        print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
        feature = int(input("Choose one of the feature space from above: "))
        k = int(input("Enter k value: "))
        print("Select one of the dimensionality reduction methods: ")
        print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
        dim_red_method = int(input("Choose one from above: "))

        feature_names = ['color_moment', 'hog', 'layer3', 'avgpool', 'fc']
        dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

        feature_space = []
        user_feature = feature_names[feature - 1]
        method = dim_red_names[dim_red_method-1]
        descriptor = (k, user_feature, method)
    elif ls == 2:
        print("select one of the features: ")
        print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
        feature = int(input("Choose one of the feature space from above: "))
        k = int(input("Enter k value: "))
        feature_names = ['color_moment', 'hog', 'layer3', 'avgpool', 'fc']
        user_feature = feature_names[feature - 1]
        descriptor = (k, user_feature)
    elif ls == 3:
        print("select one of the features: ")
        print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
        feature = int(input("Choose one of the feature space from above: "))
        k = int(input("Enter k value: "))
        print("Select one of the dimensionality reduction methods: ")
        print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
        dim_red_method = int(input("Choose one from above: "))

        feature_names = ['color_moment', 'hog', 'layer3', 'avgpool', 'fc']
        dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

        feature_space = []
        user_feature = feature_names[feature - 1]
        method = dim_red_names[dim_red_method-1]
        descriptor = (k, user_feature, method)
    elif ls == 4:
        print("select one of the features: ")
        print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
        feature = int(input("Choose one of the feature space from above: "))
        k = int(input("Enter k value: "))
        print("Select one of the dimensionality reduction methods: ")
        print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
        dim_red_method = int(input("Choose one from above: "))

        feature_names = ['color_moment', 'hog', 'layer3', 'avgpool', 'fc']
        dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

        feature_space = []
        user_feature = feature_names[feature - 1]
        method = dim_red_names[dim_red_method-1]
        descriptor = (k, user_feature, method)
    else:
        print("Invalid input")
        exit()
else:
    print("Enter the feature model: ")
    print("1. Color Moment\n2. HoG\n3. Layer3\n4. AvgPool\n5. FC")
    feature = int(input("Choose one of the feature space from above: "))
    feature_names = ['color_moment', 'hog', 'layer3', 'avgpool', 'fc']
    descriptor = feature_names[feature - 1]

n = int(input("Enter the value of n: "))
m = int(input("Enter the value of m: "))
label = int(input("Enter the label: "))

print(f"Output for {descriptor}")
print(main(descriptor, n, m, label, is_latent_space))