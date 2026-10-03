import torch.nn as nn
from torchvision import models

MODEL_NAMES = ('cnn', 'resnet18', 'vgg16', 'inception_v3')

def build_model(name, pretrained=False):
    if name == 'cnn':
        layers = []
        channels = 3
        for output_channels in (16, 32, 64):
            layers += [nn.Conv2d(channels, output_channels, 3, padding=1),
                       nn.BatchNorm2d(output_channels), nn.ReLU(), nn.MaxPool2d(2)]
            channels = output_channels
        return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Dropout(.3), nn.Linear(64, 1))
    if name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    elif name == 'inception_v3':
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None,
            aux_logits=True, transform_input=True, init_weights=False)
        # Keep the same ImageNet-to-Inception input mapping during training and
        # checkpoint reload, including when weights=None avoids a download.
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 1)
    else:
        raise ValueError(f'Unknown model {name!r}')
    return model

def image_size(name):
    return 299 if name == 'inception_v3' else 224
