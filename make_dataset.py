from crop_feature import crop_feature
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import cv2
import glob
import h5py
import argparse

parser = argparse.ArgumentParser(description="make Dataset")
parser.add_argument("--dataset", type=str)
parser.add_argument("--featureType", type=str)
parser.add_argument("--scaleFactor", type=int)
parser.add_argument("--batchSize", type=int, default=16)
parser.add_argument("--threads", type=int, default=3)

# dataset, feature_type, scale_factor, batch_size, num_workers
def main():
    opt = parser.parse_args()

    dataset = opt.dataset
    feature_type = opt.featureType
    scale_factor = opt.scaleFactor
    batch_size = opt.batchSize
    num_workers = opt.threads

    print_message = True
    dataset = dataset+"/LR_2"
    image_path = os.path.join(dataset, feature_type)
    image_list = os.listdir(image_path)
    input = list()
    label = list()

    for image in image_list:
        origin_image = Image.open(os.path.join(image_path,image))
        label.append(np.array(origin_image).astype(float))
        image_cropped = crop_feature(os.path.join(image_path, image), feature_type, scale_factor, print_message)
        print_message = False
        # bicubic interpolation
        reconstructed_features = list()
        print("crop is done.")
        for crop in image_cropped:
            w, h = crop.size
            bicubic_interpolated_image = crop.resize((w//scale_factor, h//scale_factor), Image.BICUBIC)
            bicubic_interpolated_image = bicubic_interpolated_image.resize((w,h), Image.BICUBIC) # 다시 원래 크기로 키우기
            reconstructed_features.append(np.array(bicubic_interpolated_image).astype(float))
        input.append(concatFeatures(reconstructed_features, image, feature_type))

    print("concat is done.")
    if len(input) == len(label):
        save_h5(input, label, 'data/train_{}.h5'.format(feature_type))
        print("saved..")
    else:
        print(len(input), len(label), "이 다릅니다.")


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

def save_h5(sub_ip, sub_la, savepath):
    if not os.path.exists("data/"):
        os.makedirs("data/")    
    
    path = os.path.join(os.getcwd(), savepath)
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('input', data=sub_ip)
        hf.create_dataset('label', data=sub_la)

if __name__ == "__main__":
    main()