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

class_prob_dict = {}

for i in range(len(class_prob)):
    class_prob_dict[class_prob[i]] = i

cp_keys = sorted(list(class_prob_dict.keys()), reverse=True)

print("Top K matching labels for image ID " + str(query_image))
print("| Label | Score |")
i = 0

for i in range(10):
    print("| " + str(class_prob_dict[cp_keys[i]]) +
          " | " + str(cp_keys[i].item()) + " |")
