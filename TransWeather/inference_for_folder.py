import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as utils
import os
import numpy as np
import random
from transweather_model import Transweather
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters for Inference on a folder')

parser.add_argument('-model_path', help='Path of the model', type=str)
parser.add_argument('-folder_path', help='Path of the folder', type=str)
parser.add_argument('-save_path', help='Path to save the output image', type=str)

args = parser.parse_args()

model_path = os.path.abspath(args.model_path)
folder_path = os.path.abspath(args.folder_path)
save_path = os.path.abspath(args.save_path)

os.chdir('./TransWeather')

# --- dataset --- #
class LoadData(data.Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.input_names = sorted(os.listdir(data_dir))
        self.data_dir = data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        input_img = Image.open(os.path.join(self.data_dir,input_name))
        
        # Resizing image in the multiple of 16"
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
        
        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_im = transform_input(input_img)

        return input_im, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)



def save_image(pred_image, image_name, exp_name):
    pred_image_images = torch.split(pred_image, 1, dim=0)
    batch_num = len(pred_image_images)
    
    for ind in range(batch_num):
        image_name_1 = image_name[ind].split('/')[-1]
        utils.save_image(pred_image_images[ind], '{}/{}'.format(save_path,image_name_1))


def inference_net(net, data_loader, device, exp_name):

    for batch_id, data in enumerate(data_loader):

        with torch.no_grad():
            input_im, imgid = data
            input_im = input_im.to(device)
            pred_image = net(input_im)

        # --- Save image --- #
        save_image(pred_image, imgid, exp_name)
        print('Saving ',imgid[0])

exp_name = 'model_weights'

#set seed
seed = None
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
data_dir = folder_path

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- data loader --- #
data_loader = DataLoader(LoadData(data_dir), batch_size=1, shuffle=False, num_workers=8)

# --- Define the network --- #

net = Transweather().cuda()


net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
net.load_state_dict(torch.load('{}/{}/best'.format(model_path,exp_name)))

# --- Use the evaluation model --- #
net.eval()

print('--- Inference starts! ---')

inference_net(net, data_loader, device, exp_name)
