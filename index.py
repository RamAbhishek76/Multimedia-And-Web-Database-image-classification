import torch, cv2, math, os, pandas, time, logging
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
collection = dbname.torchvision_caltech_101_features

caltech101 = '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101/caltech101/101_ObjectCategories'
category_names = sorted([name for name in os.listdir(caltech101)])

for category in category_names:
    images = []
    color_moments = []
    hog = []
    categories = []
    avgpool = []
    layer3 = []
    fc = []
    print(category)
    # cat_path = os.path.join(caltech101, 'accordion')
    cat_path = os.path.join(caltech101, category)

    cat_start_time = time.time()
    for image in os.listdir(cat_path):
        print(image)
        image_features = {}
        image_path = os.path.join(cat_path, image)

        # Image is being read as an multi-dim array of pixels for extracting HOG and Color moments
        img = cv2.imread(image_path)
        # Resizing the image into 300x100 size
        img = cv2.resize(img, (300, 100))
        # image is being read as JpegImageFile Class for input to torchvision transformation models
        resnet_input = Image.open(image_path)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        image_tensor = preprocess(resnet_input)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        if image_tensor.size()[1] == 3:
            images.append(category + "_" + image.split('.')[0])
            image_features["image_name"] = category + "_" + image.split('.')[0]
            categories.append(category)
            image_features["category"] = category

            #########################################
            #### Feature Calculation Starts here ####
            #########################################

            # Calculating the color moments with a custom defined func
            img_color_moment = extract_color_moment(img)
            color_moments.append(img_color_moment)
            image_features["color_moment"] = img_color_moment
            
            image_hog = extract_hog(img)
            hog.append(image_hog)
            image_features["hog"] = image_hog

            #Extracting features from resnet50 using custom function
            resnet_features = extract_from_resnet50(os.path.join(cat_path, image))

            layer3.append(resnet_features['layer3'])
            fc.append(resnet_features['fc'])
            avgpool.append(resnet_features['avgpool'])

            image_features["layer3"] = resnet_features['layer3']
            image_features["fc"] = resnet_features['fc']
            image_features["avgpool"] = resnet_features['avgpool']
            collection.insert_one(image_features)

    print({'avgpool': len(avgpool), 'layer3': len(layer3), 'fc': len(fc), 'cm': len(color_moments)})
    run_times.append({category: time.time() - cat_start_time})
    logging.warning(str(time.time() - cat_start_time) + " taken to process " + category + "images.")

out = pandas.DataFrame({'Image': images, 'Category': categories, 'Color Moment': color_moments, 'HOG': hog, 'AvgPool': avgpool, 'Layer 3': layer3, 'FC': fc })
out.to_csv('cache_out.csv')

end_time = time.time()
logging.warning("Total run time is " + str(end_time - start_time))
print('Total run-time = ', start_time - end_time)
