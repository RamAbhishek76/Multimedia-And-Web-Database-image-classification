from PIL import Image
import torchvision, torch
from torchvision import datasets, models
import torchvision.transforms as transforms

from database_connection import connect_to_mongo
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

mongo_client = connect_to_mongo()
dbnme = mongo_client.cse515_project_phase1
collection = dbnme.features

transforms = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
dataset = torchvision.datasets.Caltech101('/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101', transform=transforms)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

caltech_101 = []

for image_ID in range(8677):
    image, label = dataset[image_ID]
    numpy_image = (image * 255).byte().cpu().numpy().transpose((1, 2, 0))

    img = [cv2.resize(i, (300, 100)) for i in numpy_image]
    resnet_img = [cv2.resize(i, (224, 224)) for i in numpy_image]

    if len(img) == 3:
        color_moment = extract_color_moment(img)
        hog = extract_hog(img)
        resnet_features = extract_from_resnet(resnet_img)
        layer3 = resnet_features["layer3"]
        fc = resnet_features["fc"]
        avgpool = resnet_features["avgpool"]

        collection.insert_one({
            "image_id": image_ID,
            "image": image.numpy().tolist(),
            "target": label,
            "color_moment": color_moment,
            "hog": hog,
            "avgpool": avgpool,
            "layer3": layer3,
            "fc": fc
        })

        output_collection.insert_one(image_features)
        print(image["image_id"])
    else:
        collection.insert_one({
            "image_id": image_ID,
            "image": image.numpy().tolist(),
            "target": label,
            "color_moment": [],
            "hog": [],
            "avgpool": [],
            "layer3": [],
            "fc": []
        })

        output_collection.insert_one(image_features)
        print(str(image_ID) + " does not have 3 channels.")
    print(image_ID)