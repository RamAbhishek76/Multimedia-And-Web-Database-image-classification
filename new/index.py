import numpy, cv2
from database_connection import connect_to_mongo

from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

client = connect_to_mongo()
dbname = client.cse515_project_phase1
collection = dbname.torchvision_caltech_101_imageID_map
output_collection = dbname.caltech_101_features_with_imageIDs

for image in collection.find({"image_id": { "$gt": 1580}}):
    img = [cv2.resize(i, (300, 100)) for i in numpy.array(image["image"])]
    if len(img) == 3:
        color_moment = extract_color_moment(img)
        hog = extract_hog(img)
        resnet_features = extract_from_resnet(img)
        layer3 = resnet_features["layer3"]
        fc = resnet_features["fc"]
        avgpool = resnet_features["avgpool"]

        image_features = {
            "image_id": image["image_id"],
            "image_target": image["target"],
            "color_moment": color_moment,
            "hog": hog,
            "avgpool": avgpool,
            "layer3": layer3,
            "fc": fc
        }

        output_collection.insert_one(image_features)
        print(image["image_id"])
    else:
        print(str(image["image_id"]) + " does not have 3 channels.")