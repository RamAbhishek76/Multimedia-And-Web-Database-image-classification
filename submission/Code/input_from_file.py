from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
import torchvision


def input_from_file():
    path = str(input("Enter the image file path: "))

    image = Image.open(path)

    # Convert the image to a three-channel tensor (numpy array)
    image_tensor = np.array(image)

    # Ensure the image is in three channels (RGB)
    if len(image_tensor.shape) == 2:  # Grayscale image
        image_tensor = np.stack(
            (image_tensor, image_tensor, image_tensor), axis=-1)
    elif image_tensor.shape[2] == 1:  # Single-channel image
        image_tensor = np.concatenate(
            (image_tensor, image_tensor, image_tensor), axis=-1)

    print("Shape of the image tensor:", image_tensor.shape)
    return torch.tensor(image_tensor).permute(2, 0, 1).byte()
