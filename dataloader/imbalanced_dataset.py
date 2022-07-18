import os
import cv2
import torch
import _pickle as cPickle
from torchvision.transforms import transforms


class ImTrainloader(torch.utils.data.Dataset):
    def __init__(self, data_path, **kwargs):
        super(ImTrainloader, self).__init__()
        self.main_path = data_path
        self.data_path = os.path.join(data_path, 'train')
        self.cats = os.listdir(self.data_path)
        self.all_images = self.load_all_idx()
        self.meta_label = self.load_meta()
        self.all_samples = self.put_down_ram()
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def put_down_ram(self):
        img_list = []
        for item in self.all_images:
            img_list.append(cv2.imread(item))
        return img_list

    def load_meta(self):
        f = open(os.path.join(self.main_path, 'batches.meta'), 'rb')
        dict1 = cPickle.load(f, encoding='latin1')
        cat_dict = dict()
        for i, cat in enumerate(dict1['label_names']):
            cat_dict.update({cat: i})
        return cat_dict

    def load_all_idx(self):
        idx_list = []
        for cat in self.cats:
            imgs = os.listdir(os.path.join(self.data_path, cat))
            for image in imgs:
                idx_list.append(os.path.join(self.data_path, cat, image))
        return idx_list

    def __getitem__(self, idx):
        img = self.all_samples[idx]
        img = self.transform(img)
        label = self.meta_label[os.path.split(os.path.split(self.all_images[idx])[0])[1]]
        return img, label

    def __len__(self):
        return len(self.all_images)

