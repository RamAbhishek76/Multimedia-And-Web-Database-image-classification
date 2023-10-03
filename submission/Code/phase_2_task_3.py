import numpy as np

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features

cm = []
k = 400
for img in collection.find():
    print(img['image_id'])
    cm.append(np.array(img["color_moment"]).flatten())

cm = np.array(cm)

U, S, VT = np.linalg.svd(cm)

U_k = U[:, :k]
S_k = np.diag(S[:k])

latent_semantics = np.dot(U_k, S_k)

print(latent_semantics)
