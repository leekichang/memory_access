import torch
import torchvision.models
import torch.nn as nn


############# Hyper Parameter #############
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCH = 10
DEVICE = 'cpu'

if torch.cuda.is_available():
    DEVICE = 'cuda'

############# Model #############
vgg16 = torchvision.models.vgg16(pretrained=True)
for params in vgg16.features.parameters():
    params.requires_grad = False
vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

# print(vgg16.classifier)


############# Dataset #############

import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),])
data = torchvision.datasets.ImageFolder(root = './data/train', transform = transform)

train_data = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)
DEVICE = 'cpu'
for epoch in range(EPOCH):
    for idx, batch in enumerate(train_data):
        x, y = batch
        vgg16.to(DEVICE)
        x.to(DEVICE)
        y.to(DEVICE)
        print(vgg16(x))
        # print(x, y, sep='\n')
