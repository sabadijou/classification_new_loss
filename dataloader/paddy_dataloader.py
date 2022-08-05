import os
import torch
from PIL import Image
import numpy as np
from utils.transformers import imagenet_transformer_train

class PaddyDataLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, **kwargs):
        super(PaddyDataLoader, self).__init__()
        self.main_path = data_path
        self.data_path = os.path.join(data_path, 'train_images')
        self.cats = os.listdir(self.data_path)
        self.all_images, self.labels = self.load_all_idx()
        self.transformer = imagenet_transformer_train()

    def load_all_idx(self):
        idx_list = []
        cat_list = []
        counter = 0
        for cat in self.cats:
            imgs = os.listdir(os.path.join(self.data_path, cat))
            for image in imgs:
                cat_list.append(counter)
                idx_list.append(os.path.join(self.data_path, cat, image))
            counter += 1
        return idx_list, cat_list

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        img = Image.open(img_path)
        img = self.transformer(img)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.all_images)


# a = PaddyDataLoader(data_path=r'D:\Datasets\image\paddy')
# print(a.__getitem__(5))
