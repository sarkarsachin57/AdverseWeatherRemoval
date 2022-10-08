import os, argparse

parser = argparse.ArgumentParser(description='Hyper-parameters for Inference')

parser.add_argument('-model_path', help='Path of the model', type=str)
parser.add_argument('-image_path', help='Path of the image', type=str)
parser.add_argument('-save_path', help='Path to save the output image', type=str)

args = parser.parse_args()

model_path = os.path.abspath(args.model_path)
image_path = os.path.abspath(args.image_path)
save_path = os.path.abspath(args.save_path)

os.system(f'python ./TransWeather/inference.py -model_path {model_path} -image_path {image_path} -save_path {save_path}')

print(f'Image saved to {save_path} successfully.')
