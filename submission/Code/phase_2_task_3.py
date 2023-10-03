import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features

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
        U, S, VT = np.linalg.svd(feature_space)

        U_k = U[:, :k]
        S_k = np.diag(S[:k])

        latent_semantics = np.dot(U_k, S_k)

        print(latent_semantics)

    case 2:
        # TODO: Implement NNMF
        print("NNMF")

    case 3:
        print("LDA")
        feature_space = feature_space + abs(np.min(feature_space)) + 1
        lda = LatentDirichletAllocation(n_components=k, random_state=42)
        print(feature_space)
        lda.fit(feature_space)

        for topic_idx, topic in enumerate(lda.components_):
            print(f'Topic {topic_idx}:')
            top_integers = [str(i) for i in topic.argsort()[:-10 - 1:-1]]
            print(' '.join(top_integers))
            print()
    case 4:
        # TODO: Implement K Means
        print("KMeans")
