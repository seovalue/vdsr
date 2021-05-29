from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch
import pdb
import math
import numpy as np
import cv2


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
        self.names = os.listdir(self.hr_path)
        w, h = Image.open(self.hr_path[0]).size
        self.width = int(w / 16)
        self.height = int(h / 16)
        self.lwidth = int(self.width / self.rescale_factor) # rescale_factor만큼 크기를 줄인다.
        self.lheight = int(self.height / self.rescale_factor)
        print("lr: ({} {}), hr: ({} {})".format(self.lwidth, self.lheight, self.width, self.height))

        self.original_hr_imgs = []
        for hr in self.hr_path: # 256개의 피쳐로 나눈다.
            self.original_hr_imgs.append(hr) # 원본을 저장한다.
            hr_cropped_imgs = []
            hr_image = Image.open(hr)  # .convert('RGB')\
            for i in range(16):
                for j in range(16):
                    (left, upper, right, lower) = (
                    i * self.width, j * self.height, (i + 1) * self.width, (j + 1) * self.height)
                    crop = hr_image.crop((left, upper, right, lower))
                    hr_cropped_imgs.append(crop)
            self.hr_imgs.append(hr_cropped_imgs)

        self.final_results = []
            # hr_imgs = [[], [], [], ... ,[]] 내부에 500개의 []가 들어감.
        for i in range(0, len(self.hr_imgs)):
            hr_img = self.hr_imgs[i]  
            interpolated_images = []
            for img in hr_img:
                image = image.resize((self.lheight, self.lwidth), Image.BICUBIC)
                image = image.resize((self.height, self.width), Image.BICUBIC)
                interpolated_images.append(image)
            self.final_results.append(concatFeatures(interpolated_images, self.names[i], self.datatype))

    def __getitem__(self, idx):
        ground_truth = self.original_hr_imgs[idx] 
        final_result = self.final_results[idx] # list
        return transforms.ToTensor()(final_result), transforms.ToTensor()(ground_truth) # hr_image를 변환한 것과, 변환하지 않은 것을 Tensor로 각각 반환

    def __len__(self):
        return len(self.hr_path * 16 * 16)


def concatFeatures(features, image_name, feature_type):
    features_0 = features[:16]
    features_1 = features[16:32]
    features_2 = features[32:48]
    features_3 = features[48:64]
    features_4 = features[64:80]
    features_5 = features[80:96]
    features_6 = features[96:112]
    features_7 = features[112:128]
    features_8 = features[128:144]
    features_9 = features[144:160]
    features_10 = features[160:176]
    features_11 = features[176:192]
    features_12 = features[192:208]
    features_13 = features[208:224]
    features_14 = features[224:240]
    features_15 = features[240:256]
    
    features_new = list()
    features_new.extend([
        concat_vertical(features_0),
        concat_vertical(features_1),
        concat_vertical(features_2),
        concat_vertical(features_3),
        concat_vertical(features_4),
        concat_vertical(features_5),
        concat_vertical(features_6),
        concat_vertical(features_7),
        concat_vertical(features_8),
        concat_vertical(features_9),
        concat_vertical(features_10),
        concat_vertical(features_11),
        concat_vertical(features_12),
        concat_vertical(features_13),
        concat_vertical(features_14),
        concat_vertical(features_15)
    ])

    final_concat_feature = concat_horizontal(features_new)

    save_path = "features/LR_2/" + feature_type + "/" + image_name
    if not os.path.exists("features/"):
        os.makedirs("features/")
    if not os.path.exists("features/LR_2/"):
        os.makedirs("features/LR_2/")
    if not os.path.exists("features/LR_2/" + feature_type):
        os.makedirs("features/LR_2/" + feature_type)
    cv2.imwrite(save_path, final_concat_feature)
    
    return np.array(final_concat_feature).astype(float)

def concat_horizontal(feature):
    result = cv2.hconcat([feature[0], feature[1]])
    for i in range(2, len(feature)):
        result = cv2.hconcat([result, feature[i]])
    return result
       
def concat_vertical(feature):
    result = cv2.vconcat([feature[0], feature[1]])
    for i in range(2, len(feature)):
        result = cv2.vconcat([result, feature[i]])
    return result


def get_data_loader_test_version(data_path, feature_type, rescale_factor, batch_size, num_workers):
    full_dataset = FeatureDataset(data_path, feature_type, rescale_factor, False)
    print("dataset의 사이즈는 {}".format(len(full_dataset)))
    for f in full_dataset:
        print(type(f))


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