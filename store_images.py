import torchvision, torch, numpy, cv2
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
from database_connection import connect_to_mongo

mongo_client = connect_to_mongo()
dbnme = mongo_client.cse515_project_phase1
collection = dbnme.torchvision_caltech_101_imageID_map

transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
dataset = torchvision.datasets.Caltech101('/home/abhinavgorantla/hdd/ASU/Fall 23 - 24/CSE515 - Multimedia and Web Databases/project/caltech101', transform=transforms)
data_loader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)

caltech_101 = []

image_id = 0
img = dataset[880][0]

#plt.imshow(cv2.cvtColor(dataset[0][0].numpy(), cv2.COLOR_BGR2RGB))
#plt.imshow((numpy.squeeze(img.permute(1 , 2 , 0))))
#plt.show()
for image, target in dataset:
    collection.insert_one({
        "image_id": image_id,
        "image": image.numpy().tolist(),
        "target": target
        })
    print(image_id)
    image_id+=1

