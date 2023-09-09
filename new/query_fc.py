import torch, cv2, math, os, pandas, time, logging, json, numpy
from PIL import Image, ImageStat
from torchvision import transforms
from scipy.spatial import distance
import matplotlib.pyplot as plt
from database_connection import connect_to_mongo

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
image = collection.find_one({"image_id": 0})
image_raw = image_collection.find_one({"image_id": 0})

img = [cv2.resize(i, (300, 100)) for i in numpy.array(image_raw['image'])]
image = numpy.array(image['fc'])
if len(img) == 3:
    for document in collection.find({"image_id": {"$gt": -1}}):
        test = numpy.array(document["fc"]).flatten()
        q = image.flatten()
        d = distance.cosine(test, q)
        if(d in ress):
            ress[d].append([document["image_id"], document["image_target"]])
        else:
            ress[d] = [[document["image_id"], document["image_target"]]]
        print(str(document["image_target"]) + " " + str(d))
    keys = sorted(ress.keys())
    print(keys[:20])

    # fig = plt.figure(figsize=(10, 7))
    # fig.add_subplot(2, 6, 1)

    # image = image_collection.find_one({"image_id": 8676})
    # img = torch.tensor(numpy.array(image["image"]))

    # plt.imshow((numpy.squeeze(img.permute(1 , 2 , 0))))
    # plt.title("Query Image ID: " + str(image["image_id"]))
    for i in range(1, 11):
        print(ress[keys[i]][0][0])
        # image = image_collection.find_one({"image_id": ress[keys[i]][0][0]})
        # img = torch.tensor(numpy.array(image["image"]))

        # fig.add_subplot(2, 6, i + 1)
        # plt.imshow((numpy.squeeze(img.permute(1 , 2 , 0))))
        # plt.axis('off')
        # plt.title("Result image ID: " + str(image["image_id"]))
    
    # plt.show()
else:
    print("Image does not have 3 channels, please input an image with 3 channels(Red, Green and Blue).")

