from torchvision.models import resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from PIL import Image
import cv2, torch, numpy

def process_avgpool(avgpool):
    res = []
    avgpool = avgpool.detach().numpy().flatten()
    for i in range(0, len(avgpool) - 1, 2):
        res.append((avgpool[i] + avgpool[i+1])/2)
    return res

def process_layer3(layer3):
    res = []
    layer3 = layer3.detach().numpy()[0]
    for i in range(0, len(layer3)):
        res.append(sum(layer3[i].flatten())/(14*14))
    return res

def extract_from_resnet(img):
    #img = Image.open(img)
    img = (torch.as_tensor(img) * 255).byte().cpu().numpy().transpose((1, 2, 0))
    img = Image.fromarray(img)
    features = {
        "avgpool": "avgpool",
        "layer3": "layer3",
        "fc": "fc"
    }

    model = resnet50()

    resnet50_feature_extractor = create_feature_extractor(model, return_nodes=features)
    preprocess = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_tensor = preprocess(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    features = resnet50_feature_extractor(image_tensor)
    processed_avgpool = process_avgpool(features['avgpool'])
    processed_fc = features['fc'].detach().numpy().tolist()[0]
    processed_layer3 = process_layer3(features['layer3'])
    processed_features = {
        "avgpool": processed_avgpool,
        "layer3": processed_layer3,
        "fc": processed_fc
    }

    process_avgpool(features['avgpool'])

    return processed_features

# extract_from_resnet50('/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/caltech-101/101_ObjectCategories/accordion/image_0001.jpg')