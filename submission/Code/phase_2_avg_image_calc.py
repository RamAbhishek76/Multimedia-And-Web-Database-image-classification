from scipy.spatial import distance
import cv2
import numpy as np

from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features
avg_image_collection = db.avg_images

for i in range(19, 101):
    avg_image_features = {
        "color_moment": [],
        "hog": [],
        "avgpool": [],
        "layer3": [],
        "fc": []
    }
    print(i)
    ctr = 0
    for image in collection.find({'target': i}):
        ctr += 1
        if len(avg_image_features['color_moment']) == 0:
            avg_image_features['color_moment'] = np.array(
                image['color_moment'])
            avg_image_features['hog'] = np.array(
                image['hog'])
            avg_image_features['avgpool'] = np.array(
                image['avgpool'])
            avg_image_features['layer3'] = np.array(
                image['layer3'])
            avg_image_features['fc'] = np.array(
                image['fc'])
        else:
            avg_image_features['color_moment'] += np.array(
                image['color_moment'])
            avg_image_features['hog'] += np.array(
                image['hog'])
            avg_image_features['avgpool'] += np.array(
                image['avgpool'])
            avg_image_features['layer3'] += np.array(
                image['layer3'])
            avg_image_features['fc'] += np.array(
                image['fc'])

    if len(avg_image_features['color_moment']):
        avg_image_features['color_moment'] /= ctr
    if len(avg_image_features['hog']):
        avg_image_features['hog'] /= ctr
    if len(avg_image_features['avgpool']):
        avg_image_features['avgpool'] /= ctr
    if len(avg_image_features['layer3']):
        avg_image_features['layer3'] /= ctr
    if len(avg_image_features['fc']):
        avg_image_features['fc'] /= ctr

    if type(avg_image_features["color_moment"]) == type([]):
        avg_image_collection.insert_one({
            "target": i,
            "color_moment": avg_image_features["color_moment"],
            "hog": avg_image_features["hog"],
            "avgpool": avg_image_features["avgpool"],
            "layer3": avg_image_features["layer3"],
            "fc": avg_image_features["fc"],
        })
    else:
        avg_image_collection.insert_one({
            "target": i,
            "color_moment": avg_image_features["color_moment"].tolist(),
            "hog": avg_image_features["hog"].tolist(),
            "avgpool": avg_image_features["avgpool"].tolist(),
            "layer3": avg_image_features["layer3"].tolist(),
            "fc": avg_image_features["fc"].tolist(),
        })
