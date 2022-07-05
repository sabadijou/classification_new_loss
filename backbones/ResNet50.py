from utils.gpu import device
import torch.nn as nn
import torch


class ResNet50(nn.Module):
    def __init__(self, class_num=1000, pretrained=False):
        super(ResNet50, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'resnet50',
                                    pretrained=self.pretrained)

        # set last layer ################################
        if self.model.fc.out_features != self.class_num:
            self.model.fc = nn.Linear(2048, self.class_num)
        self.model = self.model.to(device)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out

