import os
from PIL import Image

def crop_feature(datapath, feature_type, scale_factor, print_message=False):
  data_path = datapath
  datatype = feature_type
  rescale_factor = scale_factor
  if not os.path.exists(data_path):
      raise Exception(f"[!] {data_path} not existed")

  hr_imgs = []
  w, h = Image.open(datapath).size
  width = int(w / 16)
  height = int(h / 16)
  lwidth = int(width / rescale_factor)
  lheight = int(height / rescale_factor)
  if print_message:
    print("lr: ({} {}), hr: ({} {})".format(lwidth, lheight, width, height))
  hr_image = Image.open(datapath)  # .convert('RGB')\
  for i in range(16):
      for j in range(16):
          (left, upper, right, lower) = (
          i * width, j * height, (i + 1) * width, (j + 1) * height)
          crop = hr_image.crop((left, upper, right, lower))
          crop = crop.resize((lwidth,lheight), Image.BICUBIC)
          crop = crop.resize((width, height), Image.BICUBIC)
          hr_imgs.append(crop)
      
  return hr_imgs