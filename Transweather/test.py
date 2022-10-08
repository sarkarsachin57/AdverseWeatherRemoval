import os, argparse, random, shutil, PIL
import numpy as np
from tqdm.notebook import tqdm

parser = argparse.ArgumentParser(description='Hyper-parameters for Testing TransWeather')

parser.add_argument('-model_path', help='Path of the model', type=str)
parser.add_argument('-input_path', help='Path of the input images folder of the dataset', type=str)
parser.add_argument('-target_path', help='Path of the target images folder of the dataset', type=str)
parser.add_argument('-results_path', help='Path of the folder where the results or output clear weather images will stored', type=str)

args = parser.parse_args()

src_in = os.path.abspath(args.input_path)
src_gt = os.path.abspath(args.target_path)
model_path = os.path.abspath(args.model_path)
results_path = os.path.abspath(args.results_path)

os.chdir('./TransWeather')
os.system('mkdir data')
os.chdir('./data')
os.system('mkdir test')
os.chdir('./test')
os.system('mkdir input')
os.system('mkdir gt')
os.chdir('./../..')

total_len = len(os.listdir(src_gt))

print('Total test dataset length :',total_len)

test_files = sorted(os.listdir(src_gt))
test_path_in = './data/test/input'
test_path_gt = './data/test/gt'

for i in test_files:
  shutil.copy(os.path.join(src_in,i),os.path.join(test_path_in,i))
  shutil.copy(os.path.join(src_gt,i),os.path.join(test_path_gt,i))
test_len = len(os.listdir(test_path_gt))

with open('./data/test/file_names.txt','w') as f:
  [f.writelines('./../test/input/'+file+'\n') for file in os.listdir(test_path_gt)]

os.system(f'python test.py -exp_name model_weights -model_path {model_path} -results_path {results_path}')

os.system('rm -r ./data')

print('\nModel tested successfully.')
