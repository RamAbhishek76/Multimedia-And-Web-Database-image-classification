# Functionality: Used this as a test script for querying images using FC feature
from database_connection import connect_to_mongo
from scipy.spatial import distance
import numpy
import cv2
import torchvision
import torchvision.transforms as transforms

from output_plotter import output_plotter
from color_moment import extract_color_moment
from hog import extract_hog
from resnet import extract_from_resnet

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features
transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Loading the dataset
dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=transforms, download=True)

choice = ""

while choice != 0:
    print("Caltech 101 - Image Retrieval")
    print("1. Print color moment, HoG, Resnet50-AvgPool, Resnet50-Layer3, Resnet50-FC features for an image.")
    print("2. Store feature descriptors for all images in Caltech-101")
    print("3. Print top 'k' results similar to a given input image, given its imageID")
    print("0. To exit this program")
    choice = int(input("Please select one of the above options: "))

    match choice:
        case 1:
            image_id = str(
                input("Enter image ID for which the features have to generated: "))
            print("Generating image features for image_id: " + str(image_id))

            image_details = collection.find_one({"image_id": image_id})
            print("Which image features do you want to view?")
            print("1. Image as an array")
            print("2. Image Label")
            print("3. Color Moment of the image")
            print("4. HoG feature of the image")
            print("5. AvgPool feature of the image")
            print("6. Layer3 feature of the image")
            print("7. FC feature of the image")

            feature_choice = int(
                input("Please select one of the above options: "))

            match feature_choice:
                case 1:
                    print(image_details["image"])
                case 2:
                    print(image_details["target"])
                case 3:
                    for i in range(len(image_details["color_moment"])):
                        print("Color moments for row " + str(i + 1))
                        for j in image_details['color_moment'][i]:
                            for k in j:
                                print(k)
                        print()
                case 4:
                    for i in range(len(image_details["hog"])):
                        for j in range(len(image_details["hog"][i])):
                            print("HoG for bin " + str((i + 1)*(j + 1)))
                            print(image_details["hog"][i][j])

                case 5:
                    print("Avgpool feature vector: ")
                    print(image_details["avgpool"])
                case 6:
                    print("Layer 3 feature vector: ")
                    print(image_details["layer3"])
                case 7:
                    print("FC feature vector: ")
                    print(image_details["fc"])

        case 2:
            print("Features have already been generated!")

        case 3:
            image_id = int(input("Enter image ID: "))
            k = int(input("How many similar images have to be returned"))

            img, label = dataset[image_id]
            resized_img = [cv2.resize(i, (300, 100)) for i in img.numpy()]
            # resizing the image into 224x224 to provide as input to the resnet
            resized_resnet_img = [cv2.resize(
                i, (224, 224)) for i in img.numpy()]

            # checking if the image has 3 channels
            if len(resized_img) == 3:
                color_moment = extract_color_moment(resized_img)
                hog = extract_hog(resized_img)
                resnet_features = extract_from_resnet(resized_resnet_img)
                input_image = {
                    "image_id": str(image_id),
                    "image": img.numpy().tolist(),
                    "color_moment": color_moment,
                    "hog": hog,
                    "avgpool": resnet_features["avgpool"],
                    "layer3": resnet_features["layer3"],
                    "fc": resnet_features["fc"],
                }

            input_image_color_moment = numpy.array(
                input_image["color_moment"]).flatten()
            input_image_hog = numpy.array(input_image["hog"]).flatten()
            input_image_avgpool = numpy.array(input_image["avgpool"]).flatten()
            input_image_layer3 = numpy.array(input_image["layer3"]).flatten()
            input_image_fc = numpy.array(input_image["fc"]).flatten()

            features = {
                'color_moment': {},
                'hog': {},
                'layer3': {},
                'avgpool': {},
                'fc': {}
            }

            print(
                "Which image features do you want to use to calculate similarity measure?")
            print("1. Color Moment")
            print("2. HoG feature")
            print("3. AvgPool feature")
            print("4. Layer3 feature")
            print("5. FC feature")

            feature_choice = int(
                input("Please select one of the above options: "))

            match feature_choice:
                case 1:
                    color_moment = {}
                    for image in collection.find():
                        print(image["image_id"])
                        image_color_moment = numpy.array(
                            image["color_moment"]).flatten()

                        d_cm = distance.euclidean(
                            image_color_moment, input_image_color_moment)

                        color_moment[d_cm] = image["image_id"]

                    feature_names = ['color_moment',
                                     'hog', 'avgpool', 'layer3', 'fc']
                    cm_keys = sorted(color_moment.keys())

                    output_plotter('Color Moment', input_image,
                                   color_moment, cm_keys, k)
                case 2:
                    hog = {}
                    for image in collection.find():
                        print(image["image_id"])
                        image_hog = numpy.array(image["hog"]).flatten()
                        d_hog = distance.euclidean(image_hog, input_image_hog)
                        hog[d_hog] = image["image_id"]

                    feature_names = ['color_moment',
                                     'hog', 'avgpool', 'layer3', 'fc']

                    hog_keys = sorted(hog.keys())

                    output_plotter('HoG', input_image, hog, hog_keys, k)
                case 3:
                    avgpool = {}
                    for image in collection.find():
                        print(image["image_id"])
                        image_avgpool = numpy.array(image["avgpool"]).flatten()
                        d_avgpool = distance.euclidean(
                            image_avgpool, input_image_avgpool)

                        avgpool[d_avgpool] = image["image_id"]

                    feature_names = ['color_moment',
                                     'hog', 'avgpool', 'layer3', 'fc']
                    avgpool_keys = sorted(avgpool.keys())

                    output_plotter('Avgpool', input_image,
                                   avgpool, avgpool_keys, k)
                case 4:
                    layer3 = {}
                    for image in collection.find():
                        print(image["image_id"])
                        image_layer3 = numpy.array(image["layer3"]).flatten()
                        d_layer3 = distance.euclidean(
                            image_layer3, input_image_layer3)
                        layer3[d_layer3] = image["image_id"]

                    feature_names = ['color_moment',
                                     'hog', 'avgpool', 'layer3', 'fc']
                    layer3_keys = sorted(layer3.keys())
                    output_plotter('Layer 3', input_image,
                                   layer3, layer3_keys, k)
                case 5:
                    fc = {}
                    for image in collection.find():
                        print(image["image_id"])
                        image_fc = numpy.array(image["fc"]).flatten()
                        d_fc = distance.euclidean(image_fc, input_image_fc)
                        fc[d_fc] = image["image_id"]

                    feature_names = ['color_moment',
                                     'hog', 'avgpool', 'layer3', 'fc']
                    fc_keys = sorted(fc.keys())
                    output_plotter('FC', input_image, fc, fc_keys, k)
