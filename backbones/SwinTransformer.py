from utils.gpu import device
import torch.nn as nn
import timm


class SwinTransformer(nn.Module):
    def __init__(self, class_num=1000, pretrained=False):
        super(SwinTransformer, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = timm.create_model('swin_small_patch4_window7_224',
                                       pretrained=self.pretrained)


        # set last layer ################################
        if self.model.head.out_features != self.class_num:
            self.model.head = nn.Linear(768, self.class_num)
        self.model = self.model.to(device)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out
