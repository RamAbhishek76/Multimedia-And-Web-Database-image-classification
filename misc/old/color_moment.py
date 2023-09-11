import torch, cv2, math, os, pandas, time, logging
from PIL import Image, ImageStat
from torchvision import transforms

def extract_color_moment(img):
    # Creating a 10x10 grid out of the image
    binned_img = []

    for i in range(10):
        temp = []
        for j in range(10):
            temp.append([])
        binned_img.append(temp)

    for i in range(len(img)):
        for j in range(len(img[i])):
            binned_img[int(i/10)][int(j/30)].append([i for i in img[i][j]])

    # print(len(binned_img))
    # print(len(binned_img[0]), len(binned_img[0][0]))

    sdev = []
    skw = []

    # print("----------Computing mean vals----------")
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

    # print("----------Computing std dev vals----------")
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

    # print("----------Computing skew vals----------")
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

# img_path = '/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/caltech-101/101_ObjectCategories/accordion/image_0001.jpg'
# img = cv2.imread(img_path)
# # Resizing the image into 300x100 size
# img = cv2.resize(img, (300, 100))

# print(extract_color_moment(img))