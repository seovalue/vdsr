from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import pdb
import math
import numpy as np


class FeatureDataset(Dataset):
    def __init__(self, data_path, datatype, rescale_factor, valid):
        self.data_path = data_path
        self.datatype = datatype
        self.rescale_factor = rescale_factor
        if not os.path.exists(data_path):
            raise Exception(f"[!] {self.data_path} not existed")
        if (valid):
            self.hr_path = os.path.join(self.data_path, 'valid')
            self.hr_path = os.path.join(self.hr_path, self.datatype)
        else:
            self.hr_path = os.path.join(self.data_path, 'LR_2')
            self.hr_path = os.path.join(self.hr_path, self.datatype)
        print(self.hr_path)
        self.hr_path = sorted(glob(os.path.join(self.hr_path, "*.*")))
        self.hr_imgs = []
        w, h = Image.open(self.hr_path[0]).size
        self.width = int(w / 16)
        self.height = int(h / 16)
        self.lwidth = int(self.width / self.rescale_factor) # rescale_factor만큼 크기를 줄인다.
        self.lheight = int(self.height / self.rescale_factor)
        print("lr: ({} {}), hr: ({} {})".format(self.lwidth, self.lheight, self.width, self.height))
        for hr in self.hr_path: # 256개의 피쳐로 나눈다.
            hr_image = Image.open(hr)  # .convert('RGB')\
            for i in range(16):
                for j in range(16):
                    (left, upper, right, lower) = (
                    i * self.width, j * self.height, (i + 1) * self.width, (j + 1) * self.height)
                    crop = hr_image.crop((left, upper, right, lower))
                    self.hr_imgs.append(crop)

    def __getitem__(self, idx):
        hr_image = self.hr_imgs[idx]
        transform = transforms.Compose([
            transforms.Resize((self.lheight, self.lwidth), interpolation=InterpolationMode.BICUBIC),
            transforms.Resize((self.height, self.width), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        return transform(hr_image), transforms.ToTensor()(hr_image)

    def __len__(self):
        return len(self.hr_path * 16 * 16)


def get_data_loader(data_path, feature_type, rescale_factor, batch_size, num_workers):
    full_dataset = FeatureDataset(data_path, feature_type, rescale_factor, False)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def get_training_data_loader(data_path, feature_type, rescale_factor, batch_size, num_workers):
    full_dataset = FeatureDataset(data_path, feature_type, rescale_factor, False)
    torch.manual_seed(3334)
    train_loader = torch.utils.data.DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=False)
    return train_loader


def get_infer_dataloader(data_path, feature_type, rescale_factor, batch_size, num_workers):
    dataset = FeatureDataset(data_path, feature_type, rescale_factor, True)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=False)
    return data_loader