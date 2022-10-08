import os, argparse

parser = argparse.ArgumentParser(description='Hyper-parameters for Inference')

parser.add_argument('-model_path', help='Path of the model', type=str)
parser.add_argument('-image_path', help='Path of the image', type=str)
parser.add_argument('-save_path', help='Path to save the output image', type=str)

args = parser.parse_args()

model_path = os.path.abspath(args.model_path)
image_path = os.path.abspath(args.image_path)
save_path = os.path.abspath(args.save_path)

os.chdir('./TransWeather')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transweather_model import Transweather
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize

input_img = Image.open(image_path)

wd_new,ht_new = input_img.size
if ht_new>wd_new and ht_new>1024:
    wd_new = int(np.ceil(wd_new*1024/ht_new))
    ht_new = 1024
elif ht_new<=wd_new and wd_new>1024:
    ht_new = int(np.ceil(ht_new*1024/wd_new))
    wd_new = 1024
wd_new = int(16*np.ceil(wd_new/16.0))
ht_new = int(16*np.ceil(ht_new/16.0))
input_img = input_img.resize((wd_new,ht_new), Image.ANTIALIAS)

transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
input_im = transform_input(input_img).unsqueeze(0)

device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Transweather().cuda()

net = nn.DataParallel(net, device_ids=device_ids)

net_path = os.path.join(model_path,'model_weights')
# --- Load the network weight --- #
net.load_state_dict(torch.load('{}/best'.format(net_path)))

# --- Use the evaluation model --- #
net.eval()

img_arr = net(input_im)[0].permute(1,2,0).detach().cpu().numpy() * 255
output_img = Image.fromarray(img_arr.astype('uint8'))

save_path = os.path.join(save_path,image_path.split('/')[-1])
output_img.save(save_path)
