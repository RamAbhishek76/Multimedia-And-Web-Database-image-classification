import torch, cv2, math, os, pandas, time, logging, json, numpy
from PIL import Image, ImageStat
from torchvision import transforms

from resnet import extract_from_resnet50 
from hog import extract_hog
from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog

start_time=  time.time()
run_times = []

#############################################################
#### Establish Database Connection and Collection Setup #####
#############################################################
mongo_client = connect_to_mongo()
dbname = mongo_client.cse515_project_phase1
collection = dbname.caltech_101_features

image_features = {}
ress = {}
img_path = '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/caltech-101/101_ObjectCategories/Leopards/image_0011.jpg'
img = cv2.imread(img_path)
# Resizing the image into 300x100 size
img = cv2.resize(img, (300, 100))
# image is being read as JpegImageFile Class for input to torchvision transformation models
resnet_input = Image.open(img_path)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_tensor = preprocess(resnet_input)
image_tensor = torch.unsqueeze(image_tensor, 0)
if image_tensor.size()[1] == 3:
    resnet_features = extract_from_resnet50(img_path)
    color_moment = extract_color_moment(img)
    hog = extract_hog(img)
    for document in collection.find():
        test = numpy.array(document["color_moment"]).flatten()
        q = numpy.array(color_moment).flatten()
        m = numpy.minimum(test, q).sum()
        M = numpy.maximum(test, q).sum()
        d = m/M
        if(int(d) in ress):
            ress[str(d)].append(document["image_name"])
        else:
            ress[str(d)] = [document["image_name"]]
        if d > 0.85:
            print(document["image_name"] + " " + str(d))
    
    # sorted(ress)
    keys = sorted(list(ress.keys()))
    # for i in range(10):
    # print(sorted(ress[keys[0]]))
    # print(keys)
    # print(ress)
    # print(keys[0])
    # print(keys[3])

else:
    print("Image does not have 3 channels, please input an image with 3 channels(Red, Green and Blue).")

