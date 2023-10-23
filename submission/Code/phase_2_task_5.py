from scipy.spatial import distance
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torchvision
import torch
from torchvision.transforms import transforms

from database_connection import connect_to_mongo

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)


def nnmf(feature_space, k):

    W = np.abs(np.random.randn(feature_space.shape[0], k))
    H = np.abs(np.random.randn(k, feature_space.shape[1]))

    prev_error = float('inf')
    for _ in range(100):  # Assuming a max of 100 iterations for convergence
        # Update W and H using multiplicative update rules
        WH = np.dot(W, H)
        # Adding a small value to avoid division by zero
        W *= (feature_space @ H.T) / (WH @ H.T + 1e-10)
        H *= (W.T @ feature_space) / (W.T @ WH + 1e-10)

        # Check for convergence
        error = np.linalg.norm(feature_space - np.dot(W, H))
        if abs(prev_error - error) < 1e-4:
            break
        prev_error = error

    return W


def svd(A):
    eigenvalues, V = np.linalg.eig(np.dot(A.T, A))
    eigenvalues = np.real(eigenvalues)
    V = np.real(V)

    # Step 2: Compute singular values from eigenvalues
    singular_values = np.sqrt(eigenvalues)

    # Step 3: Compute right singular vectors (V)
    V = V.T

    # Step 4: Compute left singular vectors (U)
    U = np.dot(A, V) / singular_values

    return U, singular_values, V


def k_means(feature_space, k):
    centroids = feature_space[np.random.choice(
        feature_space.shape[0], k, replace=False)]

    prev_centroids = centroids.copy()
    for _ in range(100):  # Assuming a max of 100 iterations for convergence
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(
            feature_space[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Recalculate centroids
        new_centroids = np.array(
            [feature_space[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(np.abs(new_centroids - prev_centroids) < 1e-4):
            break

        prev_centroids = new_centroids.copy()

    return new_centroids


client = connect_to_mongo()
db = client.cse515_project_phase1
features_coll = db.phase2_features
rep_images = db.phase2_representative_images
ls_collection = db.phase2_ls3

sim_matrix = np.zeros(shape=(101, 101))

print("Select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))
print("Select one of the dimensionality reduction methods: ")
print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
dim_red_method = int(input("Choose one from above: "))
ls_k = int(input("Enter the dimensionality of the latent space: "))
dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']

# Calculating the label-label similarity matrix
for i in range(101):
    for j in range(i, 101):
        print(i, j)
        i_image = list(rep_images.find({"target": {"$in": [i, j]}}))

        if len(i_image) >= 2:
            if (len(np.array(i_image[0]["feature_value"]).flatten()) == len(np.array(i_image[1]["feature_value"]).flatten())):
                sim = distance.euclidean(
                    np.array(i_image[0]["feature_value"]).flatten(), np.array(i_image[1]["feature_value"]).flatten())
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim
            else:
                sim_matrix[i][j] = 0
                sim_matrix[j][i] = 0
        else:
            sim_matrix[i][j] = 0
            sim_matrix[j][i] = 0

scaler = MinMaxScaler()
similarity_matrix_normalized = scaler.fit_transform(sim_matrix)

for i in range(len(similarity_matrix_normalized)):
    similarity_matrix_normalized[i] = 1 - similarity_matrix_normalized[i]

match dim_red_method:
    case 1:
        U, S, VT = svd(similarity_matrix_normalized)
        U_k = U[:, :ls_k]
        S_k = np.diag(S[:ls_k])

        latent_semantics = np.dot(U_k, S_k)
        latent_semantics = np.append([[i for i in range(len(latent_semantics[0]))]],
                                     latent_semantics, axis=0)

        print(latent_semantics)

        file_name = "svd_" + str(ls_k) + "_latent_semantics_" + \
            feature_names[feature - 1] + ".csv"
        np.savetxt(file_name,
                   latent_semantics, delimiter=',', fmt='%f')
        df = pd.read_csv(file_name)
        header = [i for i in range(len(df))]
        df.to_csv(file_name, index=True)

    case 2:
        print("NNMF")
        W = nnmf(similarity_matrix_normalized, ls_k)

        latent_semantics = np.append([[i for i in range(len(W[0]))]],
                                     W, axis=0)

        print(latent_semantics)

        file_name = "nnmf_" + str(ls_k) + "_label_label_similarity_" + \
            feature_names[feature - 1] + ".csv"
        np.savetxt(file_name,
                   latent_semantics, delimiter=',', fmt='%f')
        df = pd.read_csv(file_name)
        header = [i for i in range(len(df))]
        df.to_csv(file_name, index=True)
    case 3:
        print("LDA")
        similarity_matrix_normalized = similarity_matrix_normalized + \
            abs(np.min(similarity_matrix_normalized)) + 1
        lda = LatentDirichletAllocation(
            n_components=k, random_state=42, max_iter=6)
        print(similarity_matrix_normalized)
        lda.fit(similarity_matrix_normalized)

        for topic_idx, topic in enumerate(lda.components_):
            print(f'Topic {topic_idx}:')
            top_integers = [str(i) for i in topic.argsort()[:-10 - 1:-1]]
            print(' '.join(top_integers))
            print()
    case 4:
        print("KMeans")
        latent_semantics = k_means(similarity_matrix_normalized, k)
        latent_semantics_final = []

        print("Calculating Image Similarities:")
        similarities = []
        for image in features_coll.find():
            print(image["image_id"])
            d = []
            for rep_image in rep_images.find({"feature": feature_names[feature - 1]}):
                if len(rep_image["feature_value"]):
                    d.append(distance.euclidean(np.array(image[feature_names[feature - 1]]).flatten(),
                                                np.array(rep_image["feature_value"]).flatten()))
                else:
                    d.append(max(d))
            similarities.append(d)
        inn = 0
        for image in similarities:
            print(inn)
            inn += 2
            d = [distance.euclidean(np.array(image).flatten(), ls)
                 for ls in latent_semantics]
            latent_semantics_final.append(d)
            # ls_collection.insert_one(
            #     {"image_id": inn, "latent_semantic": list(d), "ls_k": k, "dim_red_method": "kmeans", "feature_space": feature_names[feature - 1]})

        latent_semantics_final = np.append([[i for i in range(len(latent_semantics_final[0]))]],
                                           latent_semantics_final, axis=0)
        file_name = "kmeans_" + str(ls_k) + "_latent_semantics_" + \
            feature_names[feature - 1] + ".csv"
        np.savetxt(file_name,
                   latent_semantics_final, delimiter=',', fmt='%f')
        df = pd.read_csv(file_name)
        header = [i for i in range(len(df))]
        df.to_csv(file_name, index=True)

for i in similarity_matrix_normalized:
    for j in i:
        print(j, " ", end="")
    print()

list_sims = sorted(list(sim_matrix.flatten()))

ctr = 1
while k != ctr:
    ctr += 1
    print(list_sims[ctr - 1])
