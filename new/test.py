import torch, cv2, math, os, pandas, time, logging, json, numpy
from PIL import Image, ImageStat
from torchvision import transforms
from scipy.spatial import distance
import matplotlib.pyplot as plt
from database_connection import connect_to_mongo

from resnet import extract_from_resnet
im_id = str(input("Enter the imageID: "))

#############################################################
#### Establish Database Connection and Collection Setup #####
#############################################################
mongo_client = connect_to_mongo()
dbname = mongo_client.cse515_project_phase1
image_collection = dbname.features
collection = dbname.features

image_features = {}
results = {}
query_image_features = collection.find_one({"image_id": im_id})
query_image_data = image_collection.find_one({"image_id": im_id})
print(query_image_features["image"])

img = [cv2.resize(i, (300, 100)) for i in numpy.array(query_image_data['image'])]

query_image_avgpool = numpy.array(query_image_features['avgpool'])
if len(img) == 3:
    for document in collection.find({}):
        print("here")
        if document["image_id"] == '0':
            numpy_image = numpy.array(document["image"] * 255)
            resnet_img = [cv2.resize(i, (224, 224)) for i in numpy_image]
            extract_from_resnet(resnet_img)
            # print(document["avgpool"])
            # print(len(document["avgpool"]))
            # print(max(document["avgpool"]))
            # print(document["avgpool"][0])
            exit()
else:
    print("Image does not have 3 channels, please input an image with 3 channels(Red, Green and Blue).")