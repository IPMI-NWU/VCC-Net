import timm.models
import torch
from torch import nn
from torchvision import models


class resnet18(nn.Module):
    def __init__(self, num_classes):
        super(resnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.resnet(img)


class resnet50(nn.Module):
    def __init__(self, num_classes):
        super(resnet50, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.resnet(img)


class resnet101(nn.Module):
    def __init__(self, num_classes):
        super(resnet101, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.resnet(img)


class densenet121(nn.Module):
    def __init__(self, num_classes):
        super(densenet121, self).__init__()
        self.net = models.densenet121(pretrained=True)
        self.net.classifier = nn.Linear(1024, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class densenet169(nn.Module):
    def __init__(self, num_classes):
        super(densenet169, self).__init__()
        self.net = models.densenet169(pretrained=True)
        self.net.classifier = nn.Linear(1024, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class mobilenet_v3(nn.Module):
    def __init__(self, num_classes):
        super(mobilenet_v3, self).__init__()
        self.net = models.mobilenet_v3_small(pretrained=True)
        self.net.classifier[3] = nn.Linear(self.net.classifier[0].out_features, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class efficientnet_b0(nn.Module):
    def __init__(self, num_classes):
        super(efficientnet_b0, self).__init__()
        self.net = timm.models.efficientnet.efficientnet_b0(pretrained=True)
        self.net.classifier = nn.Linear(1280, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class efficientnet_b4(nn.Module):
    def __init__(self, num_classes):
        super(efficientnet_b4, self).__init__()
        self.net = timm.models.efficientnet.efficientnet_b4(pretrained=True)
        print(self.net)
        self.net.classifier = nn.Linear(1792, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class efficientnet_b7(nn.Module):
    def __init__(self, num_classes):
        super(efficientnet_b7, self).__init__()
        self.net = timm.models.efficientnet.efficientnet_b7(pretrained=True)
        print(self.net)
        self.net.classifier = nn.Linear(2560, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class inception_v3(nn.Module):
    def __init__(self, num_classes):
        super(inception_v3, self).__init__()
        self.net = timm.models.inception_v3(pretrained=True)
        self.net.fc = nn.Sequential(nn.Linear(self.net.fc.in_features, num_classes))

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class inception_v4(nn.Module):
    def __init__(self, num_classes):
        super(inception_v4, self).__init__()
        self.net = timm.models.inception_v4(pretrained=True)
        self.net.last_linear = nn.Linear(1536, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class vit_s(nn.Module):
    def __init__(self, num_classes):
        super(vit_s, self).__init__()
        # self.net = timm.models.vision_transformer.vit_small_patch16_224(pretrained=True,
        #     pretrained_cfg_overlay=dict(file='models/pretrained_models/pytorch_model.bin'))
        self.net = timm.create_model('vit_small_patch16_224', num_classes=3, pretrained=True,
                   pretrained_cfg_overlay=dict(file="models/pretrained_models/pytorch_model.bin", custom_load=False))
        self.net.head = nn.Linear(384, num_classes, bias=True)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        return self.net(img)


class swin_s(nn.Module):
    def __init__(self, num_classes):
        super(swin_s, self).__init__()
        self.net = timm.models.swin_transformer.swin_small_patch4_window7_224(pretrained=True)
        self.net.head.fc = nn.Linear(768, num_classes)

    def forward(self, img):
        img = img.repeat(1, 3, 1, 1)
        output = self.net(img)
        return output
