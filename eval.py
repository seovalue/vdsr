import argparse, os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
from crop_feature import crop_feature
from PIL import Image

parser = argparse.ArgumentParser(description="PyTorch VDSR Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--dataset", default="Set5", type=str, help="dataset name, Default: Set5")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--featureType", default="p3", type=str)
parser.add_argument("--scaleFactor", default=4, type=int, help="scale factor")


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

scales = [4]

# image_list = glob.glob(opt.dataset+"/*.*") 
image_list = crop_feature(opt.dataset, opt.featureType, opt.scaleFactor)

for scale in scales:
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    for image_name in image_list:
        count += 1
        print("Processing ", image_name)
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
            im_input = im_input.cuda()
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

    print("Scale=", scale)
    print("Dataset=", opt.dataset)
    print("PSNR_predicted=", avg_psnr_predicted/count)
    print("PSNR_bicubic=", avg_psnr_bicubic/count)
    print("It takes average {}s for processing".format(avg_elapsed_time/count))
