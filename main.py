import argparse, os
from datasets import get_data_loader
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vdsr import Net
from datasets import get_training_data_loader
# from datasets import get_data_loader_test_version
# from feature_dataset import get_training_data_loader
# from make_dataset import make_dataset
import numpy as np
from dataFromH5 import Read_dataset_h5
import matplotlib.pyplot as plt
import math

# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--dataRoot", type=str)
parser.add_argument("--featureType", type=str)
parser.add_argument("--scaleFactor", type=int, default=4)
parser.add_argument("--batchSize", type=int, default=64, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=20, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")


total_loss_for_plot = list()
total_pnsr = list()

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

################## Loading Datasets ##########################
 
    print("===> Loading datasets")
    # train_set = DatasetFromHdf5("data/train.h5")
    # training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    training_data_loader = get_training_data_loader(opt.dataRoot, opt.featureType, opt.scaleFactor, opt.batchSize, opt.threads)
    # training_data_loader = make_dataset(opt.dataRoot, opt.featureType, opt.scaleFactor, opt.batchSize, opt.threads)

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(reduction='sum')

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            opt.start_epoch = weights["epoch"] + 1
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")

    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch, optimizer)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def PSNR(loss):
    psnr = 10 * np.log10(1 / (loss + 1e-10))
    # psnr = 20 * math.log10(255.0 / (math.sqrt(loss)))
    return psnr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        optimizer.zero_grad()
        input, target = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False)
        total_loss = 0
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()

        loss = criterion(model(input), target)
        total_loss += loss.item()
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

    epoch_loss = total_loss / len(training_data_loader)
    total_loss_for_plot.append(epoch_loss)
    psnr = PSNR(epoch_loss)
    total_pnsr.append(psnr)
    print("===> Epoch[{}]: loss : {:.10f} ,PSNR : {:.10f}".format(epoch, epoch_loss, psnr))
        # if iteration%100 == 0:
        #     print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))

def save_checkpoint(model, epoch, optimizer):
    model_out_path = "checkpoint/" + "model_epoch_{}_{}.pth".format(epoch, opt.featureType)
    state = {"epoch": epoch ,"model": model, "model_state_dict":model.state_dict(), "optimizer_state_dict":optimizer.state_dict(),
        "loss": total_loss_for_plot, "psnr":total_pnsr}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()