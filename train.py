import os, argparse, random, shutil, PIL
import numpy as np
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser(description='Hyper-parameters for Training TransWeather')

parser.add_argument('-model_path', help='Path of the model', type=str)
parser.add_argument('-input_path', help='Path of the input images folder of the dataset', type=str)
parser.add_argument('-target_path', help='Path of the target images folder of the dataset', type=str)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)
parser.add_argument('-batch_size', help='Set the training batch size', default=32, type=int)

args = parser.parse_args()

src_in = os.path.abspath(args.input_path)
src_gt = os.path.abspath(args.target_path)
model_path = os.path.abspath(args.model_path)
batch_size = args.batch_size
num_epochs = args.num_epochs

os.chdir('./TransWeather')
os.system('mkdir data')
os.chdir('./data')
os.system('mkdir train')
os.chdir('./train')
os.system('mkdir input')
os.system('mkdir gt')
os.chdir('./..')
os.system('mkdir test')
os.chdir('./test')
os.system('mkdir input')
os.system('mkdir gt')
os.chdir('./../..')

total_len = len(os.listdir(src_gt))

print('Total dataset length :',total_len)
print('Dataset will be splited into 80:20 ratio for training and validation.')

train_files = sorted(os.listdir(src_gt))[:int(total_len*0.8)]
train_path_in = './data/train/input'
train_path_gt = './data/train/gt'
for i in train_files:
  shutil.copy(os.path.join(src_in,i),os.path.join(train_path_in,i))
  shutil.copy(os.path.join(src_gt,i),os.path.join(train_path_gt,i))
train_len = len(os.listdir(train_path_gt))

with open('./data/train/file_names.txt','w') as f:
  [f.writelines('./../train/input/'+file+'\n') for file in os.listdir(train_path_gt)]
  
print('Traning dataset length :',train_len)

test_files = sorted(os.listdir(src_gt))[int(total_len*0.8):]
test_path_in = './data/test/input'
test_path_gt = './data/test/gt'

for i in test_files:
  shutil.copy(os.path.join(src_in,i),os.path.join(test_path_in,i))
  shutil.copy(os.path.join(src_gt,i),os.path.join(test_path_gt,i))
test_len = len(os.listdir(test_path_gt))

with open('./data/test/file_names.txt','w') as f:
  [f.writelines('./../test/input/'+file+'\n') for file in os.listdir(test_path_gt)]
  
print('Val dataset length :',test_len)

os.system('mkdir training_log')
with open('./training_log/Transweather_log.txt','w') as f:
  pass

os.system(f'python train.py -train_batch_size {batch_size} -model_path {model_path} -exp_name model_weights -epoch_start 0 -num_epochs {num_epochs}')

os.system('rm -r ./data')
os.system('rm -r ./training_log')

print('\nModel Trained successfully.')
