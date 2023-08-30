import torch, cv2, math, os, pandas
from PIL import Image, ImageStat

from resnet import extract_from_resnet50 

def extract_color_moment(img):
    # Resizing the image into 300x100 size
    resized_img = cv2.resize(img, (300, 100))

    # Creating a 10x10 grid out of the image
    binned_img = []

    for i in range(10):
        temp = []
        for j in range(10):
            temp.append([])
        binned_img.append(temp)

    for i in range(len(resized_img)):
        for j in range(len(resized_img[i])):
            binned_img[int(i/10)][int(j/30)].append([i for i in resized_img[i][j]])

    # print(len(binned_img))
    # print(len(binned_img[0]), len(binned_img[0][0]))

    sdev = []
    skw = []

    mean_vals = [[[0,0,0] for i in range(10)] for i in range(10)]
    sdev_vals = [[[0,0,0] for i in range(10)] for i in range(10)]
    skew_vals = [[[0,0,0] for i in range(10)] for i in range(10)]
    # Mean values for RGB
    for row in range(len(binned_img)):
        for col in range(len(binned_img[row])):
            for i in binned_img[row][col]:
                mean_vals[row][col][0] += i[0]
                mean_vals[row][col][1] += i[1]
                mean_vals[row][col][2] += i[2]
    # print(mean_vals)

    for r in range(len(mean_vals)):
        for c in range(len(mean_vals[r])):
            mean_vals[r][c] = [round(i/300,2) for i in mean_vals[r][c]]

    #print(mean_vals)

    # Standard Deviation values for RGB
    for row in range(len(binned_img)):
        for col in range(len(binned_img[row])):
            for i in binned_img[row][col]:
                sdev_vals[row][col][0] += round((i[0] - mean_vals[row][col][0])**2, 2)
                sdev_vals[row][col][1] += round((i[1] - mean_vals[row][col][1])**2, 2)
                sdev_vals[row][col][2] += round((i[2] - mean_vals[row][col][2])**2, 2)

    for r in range(len(sdev_vals)):
        for c in range(len(sdev_vals[r])):
            sdev_vals[r][c] = [round(math.sqrt((i/300)),2) for i in sdev_vals[r][c]]

    #print(sdev_vals)

    # Skew values for RGB
    for row in range(len(binned_img)):
        for col in range(len(binned_img[row])):
            for i in binned_img[row][col]:
                skew_vals[row][col][0] += round((i[0] - mean_vals[row][col][0])**3, 2)
                skew_vals[row][col][1] += round((i[1] - mean_vals[row][col][1])**3, 2)
                skew_vals[row][col][2] += round((i[2] - mean_vals[row][col][2])**3, 2)

    for r in range(len(sdev_vals)):
        for c in range(len(sdev_vals[r])):
            skew_vals[r][c] = [round((i/300)**(1./3.),2) for i in sdev_vals[r][c]]

    #print(skew_vals)

    color_moments = [[i for i in range(10)] for i in range(10)]

    # Color moments for RGB
    for row in range(len(skew_vals)):
        for col in range(len(skew_vals[row])):
            color_moments[row][col] = [[mean_vals[row][col][i], sdev_vals[row][col][i], skew_vals[row][col][i]] for i in range(3)]

    return color_moments


caltech101 = '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/caltech-101/101_ObjectCategories/'

for category in os.listdir(caltech101):
    images = []
    color_moments = []
    categories = []
    avgpool = []
    layer3 = []
    fc = []
    print(category)
    cat_path = os.path.join(caltech101, category)
    for image in os.listdir(cat_path):
        img = cv2.imread(os.path.join(cat_path, image))
        print(image)
        images.append(category + image.split('.')[0])
        categories.append(category)
        
        
        color_moments.append(extract_color_moment(img))
        
        #Extracting features from resnet50 using custom function
        resnet_features = extract_from_resnet50(os.path.join(cat_path, image))
        layer3.append(resnet_features['layer3'].detach())
        fc.append(resnet_features['fc'].detach())
        avgpool.append(resnet_features['avgpool'].detach())
    print({'avgpool': len(avgpool), 'layer3': len(layer3), 'fc': len(fc), 'cm': len(color_moments)})
    out = pandas.DataFrame({'Image': images, 'Category': categories, 'Color Moment': color_moments, 'AvgPool': avgpool, 'Layer 3': layer3, 'FC': fc })
    existing = pandas.read_csv('cache_out.csv')
    pandas.concat([existing, out], axis=0)
    existing.to_csv('cache_out.csv')