from typing import List

import torch
import torch.nn as nn
from torchvision import models


class FinetuneNet(nn.Module):
    """Based on https://github.com/meliketoy/fine-tuning.pytorch"""

    def __init__(self,
                 net_type: str = 'vggnet',
                 depth: int = 11,
                 widths: List[int] = [265, 128]):
        super(FinetuneNet, self).__init__()
        self.net_type = net_type
        self.depth = depth
        self.use_gpu = torch.cuda.is_available()
        model_ft, name = self._network(net_type, depth)

        print(f'| Add features of size {widths})')
        if (net_type == 'alexnet' or net_type == 'vggnet'):
            num_ftrs = model_ft.classifier[6].in_features
            feature_model = list(model_ft.classifier.children())
            feature_model = self._add_new_layers(num_ftrs, feature_model,
                                                 widths)
            model_ft.classifier = nn.Sequential(*feature_model)
        elif (net_type == 'resnet'):
            num_ftrs = model_ft.fc.in_features
            feature_model = list([model_ft.fc])
            feature_model = self._add_new_layers(num_ftrs, feature_model,
                                                 widths)
            model_ft.fc = nn.Sequential(*feature_model)

        self.model_ft = model_ft
        self.name = name

        if self.use_gpu:
            self.model_ft.cuda()

    def _add_new_layers(self, num_ftrs, feature_model: list, widths):
        feature_model.pop()
        if len(widths) == 2:
            feature_model.append(nn.Linear(num_ftrs, widths[0]))
            feature_model.append(nn.BatchNorm1d(widths[0]))
            feature_model.append(nn.ReLU(inplace=True))
            feature_model.append(nn.Linear(widths[0], widths[1]))
        elif len(widths) == 1:
            feature_model.append(nn.Linear(num_ftrs, widths[0]))
        return feature_model

    def _network(self, net_type, depth):
        if (net_type == 'alexnet'):
            net = models.alexnet(pretrained=True)
            name = 'alexnet'
        elif (net_type == 'vggnet'):
            if (depth == 11):
                net = models.vgg11(pretrained=True)
            elif (depth == 13):
                net = models.vgg13(pretrained=True)
            elif (depth == 16):
                net = models.vgg16(pretrained=True)
            elif (depth == 19):
                net = models.vgg19(pretrained=True)
            else:
                print("VGG default is vgg11")
                net = models.vgg11(pretrained=True)
            name = f'vgg-{depth}'
        elif (net_type == 'resnet'):
            if (depth == 18):
                net = models.resnet18(True)
            elif (depth == 34):
                net = models.resnet34(True)
            elif (depth == 50):
                net = models.resnet50(True)
            elif (depth == 101):
                net = models.resnet101(True)
            name = f'resnet-{depth}'
        else:
            print(
                "Error : Network should be either [alexnet / squeezenet / vggnet / resnet]")
            print("VGG default is vgg11")
            net = models.vgg11(pretrained=True)
            name = f'vgg11'
        return net, name

    def forward(self, x):
        return self.model_ft(x)
