import torch, cv2, math, os, pandas, time, logging, json
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

    # image_features["image_name"] = img_path.split('/')[-1].split('.')[0]

    # #########################################
    # #### Feature Calculation Starts here ####
    # #########################################

    # # Calculating the color moments with a custom defined func
    # img_color_moment = extract_color_moment(img)
    # image_features["color_moment"] = img_color_moment
    
    # image_hog = extract_hog(img)
    # image_features["hog"] = image_hog

    # #Extracting features from resnet50 using custom function
    # resnet_features = extract_from_resnet50(img_path)

    # image_features["layer3"] = resnet_features['layer3']
    # image_features["fc"] = resnet_features['fc']
    # image_features["avgpool"] = resnet_features['avgpool']
    # print(image_features)

    # c = 0
    # for document in collection.find():
    #     test = document["color_moment"]
    #     q = image_features["color_moment"]
    #     d = 0
    #     for i in range(len(test)):
    #         for j in range(len(test[i])):
    #             for k in range(len(test[i][j])):
    #                 d += sum([(test[i][j][k][0] - q[i][j][k][0])**2])
    #                 if((str(int(d**(0.5)))) in ress):
    #                     ress[str(int(d**(0.5)))].append(document["image_name"])
    #                 else:
    #                     ress[str(int(d**(0.5)))] = [document["image_name"]]
    #     print(document["image_name"])
    # sorted(ress)
    # keys = list(ress.keys())
    # # for i in range(10):
    # print(ress[keys[0]])
    # print(keys[0])
    # print(keys[1])
    
    # with open('convert.txt', 'w') as convert_file:
    #     convert_file.write(json.dumps(ress))
    resnet_features = extract_from_resnet50(img_path)
    for document in collection.find():
        test = document["avgpool"]
        q = resnet_features["avgpool"]
        d = 0
        for i in range(len(test)):
            d += (test[i] - q[i])**2

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