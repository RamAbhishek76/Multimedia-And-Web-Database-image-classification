import numpy as np

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
avg_coll = db.avg_images
features_coll = db.features
rep_images = db.rep_images

average_images = {}
representative_images = {}

label = 10

for i in range(19, 101):
    average_image = avg_coll.find_one({"target": i})
    mse_values = {}
    for image in features_coll.find({"target": i}):
        mse = np.mean(
            (np.array(image['fc']) - np.array(average_image['fc'])) ** 2)
        mse_values[mse] = image['image_id']
        # Select the image with the minimum mean squared error
    mse_keys = list(mse_values.keys())
    sorted(mse_keys)
    # print(mse_values[mse_keys[0]], i)
    print(i)
    if len(mse_keys) > 0:
        rep_images.insert_one(
            {"target": i, "rep_image_id": mse_values[mse_keys[0]]})
    else:
        rep_images.insert_one(
            {"target": i, "rep_image_id": 22})
