from utils.gpu import device
import torch.nn as nn
import timm


class EfficientnetV2(nn.Module):
    def __init__(self, class_num=1000, pretrained=False):
        super(EfficientnetV2, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = timm.create_model('efficientnetv2_rw_t',
                                       pretrained=self.pretrained)


        # set last layer ################################
        if self.model.classifier.out_features != self.class_num:
            self.model.classifier = nn.Linear(1024, self.class_num)
        self.model = self.model.to(device)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out
