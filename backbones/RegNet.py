import torchvision
from utils.gpu import device
import torch.nn as nn
import torch


class Regnet(nn.Module):
    def __init__(self, class_num=1000, pretrained=True):
        super(Regnet, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained

        # load model #####################################
        self.model = torch.hub.load('facebookresearch/swag',
                                    'regnety_32gf',
                                    pretrained=self.pretrained)
        # set last layer ################################
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classification_head = nn.Linear(3712, 10, True)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        for name, params in self.model.named_parameters():
            if not ('block4' in name):
                params.required_grad = False


    def forward(self, x):
        out = self.model(x)
        out = self.avg(out)
        out = self.flatten(out)
        out = self.classification_head(out)
        return out

# model = Regnet(class_num=10, pretrained=False).cuda()
# print(model.parameters)
# from PIL import Image
# from torchvision.transforms import transforms
# img = Image.open(r'D:\Datasets\image\paddy\train_images\downy_mildew\100170.jpg', mode='r')
# t = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=[320, 240])])
# img = t(img)
# img = model(img.unsqueeze(0).cuda())
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# img = F.interpolate(img.detach().cpu(), size=(320, 240), mode='bicubic')
# plt.imshow(img[0][0].numpy(), cmap='jet')
# plt.show()
