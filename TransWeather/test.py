import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from val_data_functions import ValData
from utils import validation, validation_val
import os
import numpy as np
import random
from transweather_model import Transweather

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-model_path', help='Path of the model', default='.', type=str)
parser.add_argument('-results_path', help='Path of the folder where the results or output clear weather images will stored', type=str)

args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name
model_path = args.model_path
results_path = args.results_path

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

# --- Set category-specific hyper-parameters  --- #
val_data_dir = './data/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# --- Validation data loader --- #

val_filename = 'file_names.txt' ## This text file should contain all the names of the images and must be placed in ./data/test/ directory

val_data_loader = DataLoader(ValData(val_data_dir,val_filename), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Define the network --- #

net = Transweather().cuda()


net = nn.DataParallel(net, device_ids=device_ids)


# --- Load the network weight --- #
net.load_state_dict(torch.load('{}/{}/best'.format(model_path,exp_name)))

# --- Use the evaluation model in testing --- #
net.eval()
category = "test"

if os.path.exists(results_path)==False:     
    os.makedirs(results_path)   


print('--- Testing starts! ---')
start_time = time.time()
val_psnr, val_ssim = validation_val(net, val_data_loader, device, exp_name,category, save_tag=True, results_path = results_path)
end_time = time.time() - start_time
print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(end_time))
