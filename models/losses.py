import torch
import torch.nn as nn
from torchvision import models

class GANLoss(nn.Module):
        
    def __init__(self, use_lsgan=True):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

def __call__(self, pred, target_is_real):
    if isinstance(pred, list):
        loss = 0
        for p in pred:
            target = self.real_label if target_is_real else self.fake_label
            loss += self.loss(p, target.expand_as(p))
        return loss
    else:
        target = self.real_label if target_is_real else self.fake_label
        return self.loss(pred, target.expand_as(pred))


class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad=False
        self.criterion = nn.L1Loss()
        self.layer_ids = [3,8,15,22]

def forward(self, x, y):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device).view(1,3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(x.device).view(1,3,1,1)
    x = (x - mean)/std
    y = (y - mean)/std
    loss=0
    xi = x; yi=y
    for i,layer in enumerate(self.vgg):
        xi = layer(xi)
        yi = layer(yi)
        if i in self.layer_ids:
            loss += self.criterion(xi, yi.detach())
    return loss