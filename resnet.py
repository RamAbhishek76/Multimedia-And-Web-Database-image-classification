from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import cv2, torch

# img = Image.open('/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/caltech-101/101_ObjectCategories/accordion/image_0001.jpg')
def extract_from_resnet50(image_path):
    img = Image.open(image_path)

    features = {
        "avgpool": "avgpool",
        "layer3": "layer3",
        "fc": "fc"
    }

    model = resnet50()

    resnet50_feature_extractor = create_feature_extractor(model, return_nodes=features)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    # print(resnet50_feature_extractor(image_tensor)['layer3'])
    return resnet50_feature_extractor(image_tensor)