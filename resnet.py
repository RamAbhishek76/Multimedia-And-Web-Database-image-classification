from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import cv2, torch

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
    # print(image_tensor.size()[2])
    # print(resnet50_feature_extractor(image_tensor)['layer3'])
    return resnet50_feature_extractor(image_tensor)