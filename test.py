import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),])
data = torchvision.datasets.ImageFolder(root = './data/train', transform = transform)

train_data = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)

for idx, batch in enumerate(train_data):
    x, y = batch
    print(x.shape, y.shape)
#print(train_data.batch)

