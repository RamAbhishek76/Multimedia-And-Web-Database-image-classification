import torch
import torchvision
from torchvision import models, transforms
from PIL import Image

model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

query_image = int(input("Enter the imageID of the query image:"))
k = int(input("Enter the k value: "))

dataset = torchvision.datasets.Caltech101(
    'D:\ASU\Fall Semester 2023 - 24\CSE515 - Multimedia and Web Databases', transform=preprocess, download=True)
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=4, shuffle=True, num_workers=8)

input_image = dataset[query_image][0].unsqueeze(0)

with torch.no_grad():
    resnet_output = model(input_image)

class_prob = torch.nn.functional.softmax(resnet_output[0], dim=0)

print(len(class_prob))
predicted_class_index = torch.argmax(class_prob)
print(predicted_class_index)
for i in range(k):
    print(i, class_prob[i])
