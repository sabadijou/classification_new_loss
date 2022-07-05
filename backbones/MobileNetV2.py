from utils.gpu import device
import torch.nn as nn
import torch


class MobileNetV2(nn.Module):
    def __init__(self, class_num=1000, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'mobilenet_v2',
                                    pretrained=self.pretrained)

        # set last layer ################################
        if self.model.classifier[1].out_features != self.class_num:
            self.model.classifier[1] = nn.Linear(1280, self.class_num)
        self.model = self.model.to(device)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out
