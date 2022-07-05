from utils.gpu import device
import torch.nn as nn
import torch


class Vgg19(nn.Module):
    def __init__(self, class_num=1000, pretrained=False):
        super(Vgg19, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = torch.hub.load('pytorch/vision:v0.10.0',
                                    'vgg19',
                                    pretrained=self.pretrained)

        # set last layer ################################
        if self.model.classifier[6].out_features != self.class_num:
            self.model.classifier[6] = nn.Linear(4096, self.class_num)
        self.model = self.model.to(device)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out


