import cv2, numpy, torch

def extract_hog(img):
    numpy_image = (torch.as_tensor(img) * 255).byte().cpu().numpy().transpose((1, 2, 0))
    gray_img = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2GRAY)

    gray_img = cv2.copyMakeBorder(gray_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    
    gx = [[i for i in range(300)] for j in range(100)]
    gy = [[i for i in range(300)] for j in range(100)]

    for i in range(1, len(gray_img) - 1):
        for j in range(1, len(gray_img[i]) - 1):
            gx[i - 1][j - 1] = -1*gray_img[i][j - 1] + 1*gray_img[i][j + 1]
            gy[i - 1][j - 1] = -1*gray_img[i - 1][j] + 1*gray_img[i + 1][j]

    # Magnitude matrix of the gradients
    g = [[i for i in range(300)] for j in range(100)]

    for i in range(0, len(gx)):
        for j in range(0, len(gx[0])):
            g[i][j] = (gx[i][j]**2 + gy[i][j]**2)**(0.5)

    # Angles matrix of gradients
    atan = [[i for i in range(300)] for j in range(100)]

    for i in range(0, len(gx)):
        for j in range(0, len(gx[0])):
            if(gx[i][j] == 0 and gy[i][j] == 0):
                atan[i][j] = 0
            elif(gx[i][j] == 0 or gy[i][j] == 0):
                atan[i][j] = 90
            else:
                atan[i][j] = numpy.rad2deg(numpy.arctan(gy[i][j]/gx[i][j]))
            # print(str(gy[i][j]) + " " + str(gx[i][j]) + " " + str(gy[i][j]/gx[i][j]), end=" ")
            # print(atan[i][j], end=" ")
        # print("\n")
    
    res = [[[0 for i in range(9)] for j in range(10)] for k in range(10)]

    for i in range(len(atan)):
        for j in range(len(atan[i])):
            if(atan[i][j] >= 0):
                res[int(i/10)][int(j/30)][int(atan[i][j]/40)] += g[i][j]
            else:
                res[int(i/10)][int(j/30)][int((360 + atan[i][j])/40)] += g[i][j]
    return res