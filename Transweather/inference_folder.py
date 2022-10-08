import os, argparse

parser = argparse.ArgumentParser(description='Hyper-parameters for Inference')

parser.add_argument('-model_path', help='Path of the model', type=str)
parser.add_argument('-folder_path', help='Path of the folder', type=str)
parser.add_argument('-save_path', help='Path to save the output image', type=str)

args = parser.parse_args()

model_path = os.path.abspath(args.model_path)
folder_path = os.path.abspath(args.folder_path)
save_path = os.path.abspath(args.save_path)

image_names = sorted(os.listdir(folder_path))
print(f'Total number of images {len(image_names)}')

os.system(f'python ./TransWeather/inference_for_folder.py -model_path {model_path} -folder_path {folder_path} -save_path {save_path}')

print(f'All images saved in {save_path} successfully.')
