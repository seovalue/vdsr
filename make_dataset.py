from crop_feature import crop_feature
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2

def make_dataset(dataset, feature_type, scale_factor, batch_size, num_workers):
    dataset = dataset+"/LR_2"
    image_path = os.path.join(dataset, feature_type)
    image_list = os.listdir(image_path)
    full_dataset = list()
    for image in image_list:
        image_cropped = crop_feature(os.path.join(image_path, image), feature_type, scale_factor)
        # bicubic interpolation
        reconstructed_features = list()
        for crop in image_cropped:
            w, h = crop.size
            bicubic_interpolated_image = crop.resize((w//scale_factor, h//scale_factor), Image.BICUBIC)
            bicubic_interpolated_image = bicubic_interpolated_image.resize((w,h), Image.BICUBIC) # 다시 원래 크기로 키우기
            reconstructed_features.append(np.array(bicubic_interpolated_image).astype(float))
        full_dataset.append(concatFeatures(reconstructed_features, image, feature_type))
    
    torch.manual_seed(3334)
    return torch.utils.data.DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, pin_memory=False)

def concatFeatures(features, image_name, feature_type):
    print("features size --> ", len(features))
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
    print("saved feature in {}".format(save_path))

    return final_concat_feature

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