# Implement a program which, given (a) a query label, l (b) a user selected feature space, and (c) positive integer
# k, identifies and visualizes the most relevant k images for the given label l, along with their scores, under the selected
# feature space.
from scipy.spatial import distance
import numpy as np

from output_plotter import task_2_output_plotter
from database_connection import connect_to_mongo

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features
rep_collection = db.rep_images

query_label = int(input("Enter the query label: "))
print("select one of the features: ")
print("1. Color Moment\n2.HoG\n3. Layer3\n4. AvgPool\n5. FC")
feature = int(input("Choose one of the feature space from above: "))
k = int(input("Enter k value: "))

feature_names = ['color_moment',
                 'hog', 'layer3', 'avgpool', 'fc']

# input_image = rep_collection.find_one({"target": query_label})
# print(input_image["rep_image_id"])
input_image = collection.find_one({"target": query_label})
print(input_image)
input_image_feature = input_image[feature_names[feature - 1]]

feature_ranks = {}
for image in collection.find():
    print(image["image_id"])
    image_feature = np.array(
        image[feature_names[feature - 1]]).flatten()

    d_cm = distance.euclidean(
        image_feature, np.array(input_image_feature).flatten())

    feature_ranks[d_cm] = image["image_id"]

cm_keys = sorted(feature_ranks.keys())

task_2_output_plotter(feature_names[feature - 1].capitalize(), query_label,
                      feature_ranks, cm_keys, k)
