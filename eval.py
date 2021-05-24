import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from crop_feature import crop_feature
from PIL import Image
import cv2
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--featureType", default="p3", type=str)
parser.add_argument("--scaleFactor", default=4, type=int, help="scale factor")
parser.add_argument("--singleImage", type=str, default="N", help="if it is a single image, enter \"y\"")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def concatFeatures(features):
    print("features --> ", features)
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
    features_new.append(
        concat_horizontal(features_0),
        concat_horizontal(features_1),
        concat_horizontal(features_2),
        concat_horizontal(features_3),
        concat_horizontal(features_4),
        concat_horizontal(features_5),
        concat_horizontal(features_6),
        concat_horizontal(features_7),
        concat_horizontal(features_8),
        concat_horizontal(features_9),
        concat_horizontal(features_10),
        concat_horizontal(features_11),
        concat_horizontal(features_12),
        concat_horizontal(features_13),
        concat_horizontal(features_14),
        concat_horizontal(features_15)
    )

def concat_horizontal(feature):
    result = cv2.hconcat([feature[0], feature[1]])
    for i in range(2, len(feature) - 1):
        result = cv2.hconcat([result, feature[i]])
    plt.imshow(result)
    return result
       

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

scales = [opt.scaleFactor]

# image_list = glob.glob(opt.dataset+"/*.*") 
if opt.singleImage == "Y" :
    # image_list = crop_feature(opt.dataset, opt.featureType, opt.scaleFactor)
    image_list = opt.dataset
else:
    image_path = os.path.join(opt.dataset, opt.featureType)
    image_list = os.listdir(image_path)


for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
        image_name_cropped = crop_feature(os.path.join(opt.dataset, image_name), opt.featureType, opt.scaleFactor)
        features = []
        for image in image_name_cropped:
            count += 1
            f_gt = image_name
            w, h = image_name.size 
            f_bi = image_name.resize((w//scale,h//scale), Image.BICUBIC)
            f_bi = f_bi.resize((w,h), Image.BICUBIC)

            f_gt = np.array(f_gt)
            f_bi = np.array(f_bi)    
            f_gt = f_gt.astype(float)
            f_bi = f_bi.astype(float)

            psnr_bicubic = PSNR(f_gt, f_bi,shave_border=scale)
            avg_psnr_bicubic += psnr_bicubic

            f_input = f_bi/255.
            f_input = Variable(torch.from_numpy(f_input).float()).view(1, -1, f_input.shape[0], f_input.shape[1])

            if cuda:
                model = model.cuda()
                f_input = f_input.cuda()
            else:
                model = model.cpu()

            start_time = time.time()
            SR = model(f_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            SR = SR.cpu()

            f_sr = SR.data[0].numpy().astype(np.float32)

            f_sr = f_sr * 255
            f_sr[f_sr<0] = 0
            f_sr[f_sr>255.] = 255.
            f_sr = f_sr[0,:,:]

            psnr_predicted = PSNR(f_gt, f_sr,shave_border=scale)
            avg_psnr_predicted += psnr_predicted
            features.append(f_sr)

        concatFeatures(features)
        print("Scale=", scale)
        print("Dataset=", opt.dataset)
        print("Average PSNR_predicted=", avg_psnr_predicted/count)
        print("Average PSNR_bicubic=", avg_psnr_bicubic/count)


# Show graph
# f_gt = Image.fromarray(f_gt)
# f_b = Image.fromarray(f_bi)
# f_sr = Image.fromarray(f_sr)

# fig = plt.figure(figsize=(18, 16), dpi= 80)
# ax = plt.subplot("131")
# ax.imshow(f_gt)
# ax.set_title("GT")

# ax = plt.subplot("132")
# ax.imshow(f_bi)
# ax.set_title("Input(bicubic)")

# ax = plt.subplot("133")
# ax.imshow(f_sr)
# ax.set_title("Output(vdsr)")
# plt.show()
