from scipy.spatial import distance
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
avg_coll = db.avg_images
features_coll = db.features
rep_images = db.final_representative_images

sim_matrix = np.zeros(shape=(101, 101))

print("Select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))
print("Select one of the dimensionality reduction methods: ")
print("1. SVD\n2.NNMF\n3. LDA\n4. K Means")
dim_red_method = int(input("Choose one from above: "))

dim_red_names = ["svd", "nnmf", "lda", "kmeans"]

feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']

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
        k_val = int(input("Enter k value for SVD"))
        U, S, VT = np.linalg.svd(similarity_matrix_normalized)

        U_k = U[:, :k_val]
        S_k = np.diag(S[:k_val])

        latent_semantics = np.dot(U_k, S_k)

        print("latent_semantics", latent_semantics)
        np.savetxt("svd_latent_semantics_label_label_similarity.txt",
                   latent_semantics, fmt='%f')

    case 2:
        # TODO: Implement NNMF
        print("NNMF")

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
        # TODO: Implement K Means
        print("KMeans")
for i in similarity_matrix_normalized:
    for j in i:
        print(j, " ", end="")
    print()

list_sims = sorted(list(sim_matrix.flatten()))

ctr = 1
while k != ctr:
    ctr += 1
    print(list_sims[ctr - 1])
