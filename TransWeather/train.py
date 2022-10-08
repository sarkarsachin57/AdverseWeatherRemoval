import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random

from transweather_model import Transweather
from datetime import datetime
import pytz

IST = pytz.timezone('Asia/Kolkata')

plt.switch_backend('agg')

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)
parser.add_argument('-model_path', help='Path of the model', default='.', type=str)


args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
model_path = args.model_path

print('\n')
print('-------------------------------------------')
print('|----------Training Transweather-----------|')
print('-------------------------------------------')
print('\n')

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


train_data_dir = './data/train/'
val_data_dir = './data/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Transweather()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

# --- Load the network weight --- #
if os.path.exists('{}/{}/'.format(model_path,exp_name))==False:     
    try:
      os.mkdir('{}/{}/'.format(model_path,exp_name))  
    except:
      raise FileNotFoundError(f'path not exists : {model_path}')
try:
    net.load_state_dict(torch.load('{}/{}/best'.format(model_path,exp_name)))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


# pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print("Total_params: {}".format(pytorch_total_params))
loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,'file_names.txt'), batch_size=train_batch_size, shuffle=True, num_workers=8)

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the train time 

val_data_loader = DataLoader(ValData(val_data_dir,'file_names.txt'), batch_size=val_batch_size, shuffle=False, num_workers=8)


# --- Previous PSNR and SSIM in testing --- #
net.eval()

################ Note########################

## Uncomment the other validation data loader to keep an eye on performance 
## but note that validating while training significantly increases the test time 

old_val_psnr, old_val_ssim = validation(net, val_data_loader, device, exp_name)

print('old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}\n'.format(old_val_psnr, old_val_ssim))

net.train()

for epoch in range(epoch_start,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
#-------------------------------------------------------------------------------------------------------------
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        loss = smooth_loss + lambda_loss*perceptual_loss 

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        if not (batch_id % 100):
            print('Epoch: {0}, Iteration: {1}'.format(epoch+1, batch_id))

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)

    # --- Save the network parameters --- #
    torch.save(net.state_dict(), '{}/{}/latest'.format(model_path,exp_name))
    #torch.save(net.state_dict(), '/content/drive/MyDrive/{}/latest'.format(exp_name))

    # --- Use the evaluation model in testing --- #
    net.eval()

    val_psnr, val_ssim = validation(net, val_data_loader, device, exp_name)

    one_epoch_time = time.time() - start_time
    datetime_ist = datetime.now(IST)
    
    print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name)

    # --- update the network weight --- #
    if val_psnr >= old_val_psnr:
        torch.save(net.state_dict(), '{}/{}/best'.format(model_path,exp_name))
        print('model saved\n')
        old_val_psnr = val_psnr