from PIL import Image
import numpy as np
import torch


def input_from_file():
    path = str(input("Enter the image file path: "))

    image = Image.open(path)
    image_np = np.array(image)

    image_tensor = torch.from_numpy(image_np).float()

    return image_tensor
