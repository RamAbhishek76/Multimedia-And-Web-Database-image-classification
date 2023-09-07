import torch, cv2, math, os, pandas, time, logging, json, numpy
from PIL import Image, ImageStat
from torchvision import transforms

from resnet import extract_from_resnet50 
from hog import extract_hog
from database_connection import connect_to_mongo
from color_moment import extract_color_moment

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
img_path = '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/caltech-101/101_ObjectCategories/crab/image_0012.jpg'
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
    for document in collection.find():
        test = numpy.array(document["avgpool"]).flatten()
        q = numpy.array(resnet_features["avgpool"]).flatten()
        d = 0
        tq = sum([vec[0] * vec[1] for vec in zip(test, q)])
        mt = (sum([i for i in test]))**(0.5)
        mq = (sum([i for i in q]))**(0.5)
        d = (tq)/(mt+mq)
        if((str(int(d**(0.5)))) in ress):
            ress[str(int(d**(0.5)))].append(document["image_name"])
        else:
            ress[str(int(d**(0.5)))] = [document["image_name"]]
        print(document["image_name"])
    
    # sorted(ress)
    keys = sorted(list(ress.keys()))
    # for i in range(10):
    print(sorted(ress[keys[0]]))
    print(keys[0])
    # print(ress[keys[1]])
    # print(keys[0])
    # print(keys[3])

else:
    print("Image does not have 3 channels, please input an image with 3 channels(Red, Green and Blue).")
