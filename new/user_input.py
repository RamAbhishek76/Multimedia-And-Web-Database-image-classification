from database_connection import connect_to_mongo
from scipy.spatial import distance
import numpy, torch
import matplotlib.pyplot as plt

from query_color_moment import query_color_moment
from query_hog import query_hog
from query_avgpool import query_avgpool
from query_layer3 import query_layer3
from query_fc import query_fc
from output_plotter import output_plotter

client = connect_to_mongo()
db = client.cse515_project_phase1
collection = db.features

choice = ""

while choice != "q":
    print("Caltech 101 - Image Retrieval")
    print("1. Print color moment, HoG, Resnet50-AvgPool, Resnet50-Layer3, Resnet50-FC features for an image.")
    print("2. Store feature descriptors for all images in Caltech-101")
    print("3. Print top 'k' results similar to a given input image, given its imageID")
    choice = int(input("Please select one of the above options: "))

    match choice:
        case 1:
            image_id = str(input("Enter image ID for which the features have to generated: "))
            print("Generating image features for image_id: " + str(image_id))

            image_details = collection.find_one({"image_id": image_id})
            # print(image_details)
            print("Which image features do you want to view?")
            print("1. Image as an array")
            print("2. Image Label")
            print("3. Color Moment of the image")
            print("4. HoG feature of the image")
            print("5. AvgPool feature of the image")
            print("6. Layer3 feature of the image")
            print("7. FC feature of the image")

            feature_choice = int(input("Please select one of the above options: "))

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
            image_id = str(input("Enter image ID: "))
            k = int(input("How many similar images have to be returned"))
            input_image = collection.find_one({"image_id": image_id})

            input_image_color_moment = numpy.array(input_image["color_moment"]).flatten()
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
            color_moment = {}
            hog = {}
            avgpool = {}
            layer3 = {}
            fc = {}

            for image in collection.find():
                print(image["image_id"])
                image_color_moment = numpy.array(image["color_moment"]).flatten()
                image_hog = numpy.array(image["hog"]).flatten()
                image_avgpool = numpy.array(image["avgpool"]).flatten()
                image_layer3 = numpy.array(image["layer3"]).flatten()
                image_fc = numpy.array(image["fc"]).flatten()

                d_cm = distance.cosine(image_color_moment, input_image_color_moment)
                d_hog = distance.euclidean(image_hog, input_image_hog)
                d_avgpool = distance.euclidean(image_avgpool, input_image_avgpool)
                d_layer3 = distance.euclidean(image_layer3, input_image_layer3)
                d_fc = distance.euclidean(image_fc, input_image_fc)

                color_moment[d_cm] = image["image_id"]
                hog[d_hog] = image["image_id"]
                avgpool[d_avgpool] = image["image_id"]
                layer3[d_layer3] = image["image_id"]
                fc[d_fc] = image["image_id"]

            feature_names = ['color_moment', 'hog', 'avgpool', 'layer3', 'fc']
            cm_keys = sorted(color_moment.keys())
            hog_keys = sorted(hog.keys())
            avgpool_keys = sorted(avgpool.keys())
            layer3_keys = sorted(layer3.keys())
            fc_keys = sorted(fc.keys())

            output_plotter('Color Moment', input_image, color_moment, cm_keys, k)
            output_plotter('HoG', input_image, hog, hog_keys, k)
            output_plotter('Avgpool', input_image, avgpool, avgpool_keys, k)
            output_plotter('Layer 3', input_image, layer3, layer3_keys, k)
            output_plotter('FC', input_image, fc, fc_keys, k)