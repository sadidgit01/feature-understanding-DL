import torch.nn as nn
from torchvision.models import GoogLeNet_Weights, ResNet101_Weights, googlenet, resnet101

from config import DEVICE


class ZFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def get_resnet101():
    model = resnet101(weights=ResNet101_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model


def get_googlenet():
    model = googlenet(weights=GoogLeNet_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model


def get_zfnet():
    model = ZFNet()
    model = model.to(DEVICE)
    model.eval()
    return model


def load_all_models():
    return {
        "resnet101": get_resnet101(),
        "googlenet": get_googlenet(),
        "zfnet": get_zfnet(),
    }
