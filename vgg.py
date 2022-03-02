import torch
import torchvision.models
import torch.nn as nn

vgg16 = torchvision.models.vgg16(pretrained=True)
for params in vgg16.features.parameters():
    params.requires_grad = False
vgg16.classifier[6] = nn.Linear(in_features=4096, out_features=10, bias=True)

print(vgg16.classifier)
 
