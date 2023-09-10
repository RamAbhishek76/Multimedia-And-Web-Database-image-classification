from database_connection import connect_to_mongo
from scipy.spatial import distance

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

            image_details = collection.find_one({image_id: image_id})

            print("Image details of image " + image_id + " are:")
            # "image": img.numpy().tolist(),
            # "target": label,
            # "color_moment": color_moment,
            # "hog": hog,
            # "avgpool": resnet_features["avgpool"],
            # "layer3": resnet_features["layer3"],
            # "fc": resnet_features["fc"],
            print(image_details)
        
        case 2:
            print("Features have already been generated!")
        
        case 3:
            image_id = str(input("Enter image ID: "))
            k = int(input("How many similar images have to be returned"))
            input_image = collection.find({"image_id": image_id})

            features = {
                'color_moment': {},
                'hog': {},
                'layer3': {},
                'avgpool': {},
                'fc': {}
            }

            for image in collection.find():
                features["color_moment"][distance.euclidean(input_image["color_moment"], image["color_moment"])] = (image["image_id"], image["target"])
                features["hog"][distance.euclidean(input_image["hog"], image["hog"])] = (image["image_id"], image["target"])
                features["avgpool"][distance.euclidean(input_image["avgpool"], image["avgpool"])] = (image["image_id"], image["target"])
                features["layer3"][distance.euclidean(input_image["layer3"], image["layer3"])] = (image["image_id"], image["target"])
                features["fc"][distance.euclidean(input_image["fc"], image["fc"])] = (image["image_id"], image["target"])

            feature_keys = features.keys()

            for key in feature_keys:
                feature = features[key]
                vals = sorted(features.keys())

                for i in range(k):
                    print(str(feature[vals[i]][0]) + " " + str(feature[vals[i]][1]))