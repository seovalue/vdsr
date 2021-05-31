# Feature SR

1. train

`!python main.py --dataRoot /content/drive/MyDrive/feature/HR_trainset/features --scaleFactor 4 --featureType p6 --batchSize 16 --cuda --nEpochs 20`

2. inference

`!python inference.py --cuda --model "model.pth" --dataset "/content/drive/MyDrive/feature/features/LR_2" --featureType "p3" --scaleFactor 4`

3. calculate mAP

```
# [1]
# install dependencies: 
!pip install pyyaml==5.1
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
!gcc --version
# opencv is pre-installed on colab

# [2]
# install detectron2: (Colab has CUDA 10.1 + torch 1.8)
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
import torch
assert torch.__version__.startswith("1.8")   # need to manually install torch 1.8 if Colab changes its default version
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
# exit(0)  # After installation, you need to "restart runtime" in Colab. This line can also restart runtime

# [3]
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

!python calculate_mAP.py --valid_data_path /content/drive/MyDrive/dataset/validset_100/ --model_name VDSR --loss_type MSE --batch_size 16
```