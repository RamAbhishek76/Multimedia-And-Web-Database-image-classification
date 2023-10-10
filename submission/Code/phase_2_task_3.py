import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

from database_connection import connect_to_mongo


def power_iteration(matrix, num_iterations=1000):
    """
    Perform power iteration to find the dominant singular vector of a matrix.
    """
    # Initialize a random vector (initial guess for the dominant singular vector)
    vector = np.random.rand(matrix.shape[1])

    for _ in range(num_iterations):
        # Compute the product of the matrix and the vector
        product = matrix @ vector

        # Compute the norm of the product
        norm = np.linalg.norm(product)

        # Update the vector to be the normalized product
        vector = product / norm

    return vector, norm


def svd(matrix, num_iterations=1000):
    """
    Compute the Singular Value Decomposition (SVD) of a matrix using power iteration.
    """
    # Compute A^T * A
    ata = matrix.T @ matrix

    # Find the dominant eigenvector of A^T * A (right singular vector)
    v, sigma = power_iteration(ata, num_iterations)

    # Compute the singular value
    singular_value = np.sqrt(sigma)

    # Find the corresponding left singular vector
    u = matrix @ v / singular_value

    # Stack the left singular vector, singular value, and right singular vector
    u = u.reshape(-1, 1)
    singular_value = np.array([singular_value])
    v = v.reshape(-1, 1)

    return u, singular_value, v.T


client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.phase2_features

print("select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))
print("Select one of the dimensionality reduction methods: ")
print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
dim_red_method = int(input("Choose one from above: "))

feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']
dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

feature_space = []
user_feature = feature_names[feature - 1]
for img in collection.find():
    print(img['image_id'])
    feature_space.append(np.array(img[user_feature]).flatten())

feature_space = np.array(feature_space)

match dim_red_method:
    case 1:
        print("SVD")
        # U, S, VT = np.linalg.svd(feature_space)
        U, S, VT = svd(feature_space)

        U_k = U[:, :k]
        S_k = np.diag(S[:k])

        latent_semantics = np.dot(U_k, S_k)
        latent_semantics = np.append([[i for i in range(len(latent_semantics[0]))]],
                                     latent_semantics, axis=0)

        print(latent_semantics)

        file_name = "svd_" + str(k) + "_latent_semantics_" + \
            feature_names[feature - 1] + ".csv"
        np.savetxt(file_name,
                   latent_semantics, delimiter=',', fmt='%f')
        df = pd.read_csv(file_name)
        header = [i for i in range(len(df))]
        df.to_csv(file_name, index=True)

    case 2:
        print("NNMF")
        nmf = NMF(n_components=k, init='random', random_state=42)
        W = nmf.fit_transform(feature_space)
        H = nmf.components_

        print(W)

    case 3:
        print("LDA")
        feature_space = feature_space + abs(np.min(feature_space)) + 1
        lda = LatentDirichletAllocation(
            n_components=k, random_state=42, max_iter=10)

        lda.fit(feature_space)
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            topics.append(topic)

        topics = np.append([[i for i in range(len(topics[0]))]],
                           topics, axis=0)

        print(topics)

        file_name = "lda_" + str(k) + "_latent_semantics_" + \
            feature_names[feature - 1] + ".csv"
        np.savetxt(file_name,
                   topics, delimiter=',', fmt='%f')
        df = pd.read_csv(file_name)
        header = [i for i in range(len(df))]
        df.to_csv(file_name, index=True)
    case 4:
        print("KMeans")
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(feature_space)

        print(kmeans.cluster_centers_)
