import torch, cv2, math, os, pandas, time, logging, json, numpy
from PIL import Image, ImageStat
from torchvision import transforms
from scipy.spatial import distance
import matplotlib.pyplot as plt
from database_connection import connect_to_mongo

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

query_image_hog = numpy.array(query_image_features['hog'])
if len(img) == 3:
    for document in collection.find({}):
        print(document["image_id"])
        test = numpy.array(document["hog"]).flatten()
        if len(test) > 0:
            d = distance.euclidean(test, query_image_hog.flatten())
            if(d in results):
                results[d].append([document["image_id"], document["target"]])
            else:
                results[d] = [[document["image_id"], document["target"]]]
            
            # Print the image ID along with the distance measure value
            print(str(document["target"]) + " " + str(d))
        else:
            print("The image is 1 channel")
    keys = sorted(results.keys())
    print(keys[:20])

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(2, 6, 1)

    plt.imshow((numpy.squeeze(torch.tensor(numpy.array(query_image_data["image"])).permute(1 , 2 , 0))))
    plt.title("Query Image ID: " + str(query_image_data["image_id"]))
    for i in range(1, 11):
        print(str(results[keys[i]][0][0]) + " " + str(results[keys[i]][0][1]))
        image = image_collection.find_one({"image_id": results[keys[i]][0][0]})
        img = torch.tensor(numpy.array(image["image"]))

        fig.add_subplot(2, 6, i + 1)
        plt.imshow((numpy.squeeze(img.permute(1 , 2 , 0))))
        plt.axis('off')
        plt.title("Result image ID: " + str(image["image_id"]))
    
    plt.show()
else:
    print("Image does not have 3 channels, please input an image with 3 channels(Red, Green and Blue).")