import torch, cv2, math, os, pandas, time, logging, json, numpy
from PIL import Image, ImageStat
from torchvision import transforms
from scipy.spatial import distance

from database_connection import connect_to_mongo
from color_moment import extract_color_moment

start_time=  time.time()
run_times = []

#############################################################
#### Establish Database Connection and Collection Setup #####
#############################################################
mongo_client = connect_to_mongo()
dbname = mongo_client.cse515_project_phase1
image_collection = dbname.torchvision_caltech_101_imageID_map
collection = dbname.caltech_101_features_with_imageIDs

image_features = {}
ress = {}
image = collection.find_one({"image_id": 1})
image_raw = image_collection.find_one({"image_id": 1})

img = [cv2.resize(i, (300, 100)) for i in numpy.array(image_raw['image'])]
image = numpy.array(image['color_moment'])
if len(img) == 3:
    for document in collection.find():
        test = numpy.array(document["color_moment"]).flatten()
        q = image.flatten()
        d = distance.cosine(test, q)
        if(d in ress):
            ress[d].append([document["image_id"], document["image_target"]])
        else:
            ress[d] = [[document["image_id"], document["image_target"]]]
        if d < 0.17:
            print(str(document["image_target"]) + " " + str(d))
    keys = sorted(ress.keys())
    print(keys[:20])
    for i in range(10):
        print(ress[keys[i]])

else:
    print("Image does not have 3 channels, please input an image with 3 channels(Red, Green and Blue).")

