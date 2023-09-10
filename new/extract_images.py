from PIL import Image
import torchvision, torch, cv2, numpy
from torchvision import datasets, models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

mongo_client = connect_to_mongo()
dbnme = mongo_client.cse515_project_phase1
collection = dbnme.features

transforms = transforms.Compose([
            transforms.ToTensor(),
])
dataset = torchvision.datasets.Caltech101('D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

for image_ID in range(8677):
    img, label = dataset[image_ID]

    resized_img = [cv2.resize(i, (300, 100)) for i in img.numpy()]
    resized_resnet_img = [cv2.resize(i, (224, 224)) for i in img.numpy()]

    if len(resized_img) == 3:
        color_moment = extract_color_moment(resized_img)
        hog = extract_hog(resized_img)
        resnet_features = extract_from_resnet(resized_resnet_img)
        # print(resnet_features)

        # q = { "image_id": str(image_ID) }
        # vals = { "$set": { "avgpool": resnet_features["avgpool"], "layer3": resnet_features["layer3"], "fc": resnet_features["fc"] } }

        # collection.update_one(q, vals)

        collection.insert_one({
            "image_id": str(image_ID),
            "image": img.numpy().tolist(),
            "target": label,
            "color_moment": color_moment,
            "hog": hog,
            "avgpool": resnet_features["avgpool"],
            "layer3": resnet_features["layer3"],
            "fc": resnet_features["fc"],
        })
    print(image_ID)