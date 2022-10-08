# TransWeather : Adverse Weather Image Enhancement

## Overview

Object Detection or related Computer Vision tasks are very less accurate under adverse weather conditions like heavy fog and rain compared to clear weather. 
Most of the object detection models are trained on clear weather images. So, when it comes to detecting objects on foggy or rainy images, the detectors most likely fail to detect many objects. 

### What can be the solution? 

**TransWeather** is a Transformer-based Restoration of Images Degraded by Adverse Weather Conditions. This transformer based solution is the implementation of the <a href='https://arxiv.org/pdf/2111.14813.pdf'>CVPR 2021 TransWeather Research paper</a>. Their official github repository - <a href='https://github.com/jeya-maria-jose/TransWeather'> click here </a>

![image](https://user-images.githubusercontent.com/83460431/180592123-570a7053-ebc7-4737-a7b1-b16ba181a779.png)


### What we have done?

Our initial experiment was to validate TransWeather that how good TransWeather is able to generate clear images from foggy and rainy images and how this resulted clear images helpful to make object detection more accurate. We used KITTI dataset for this experiment.<br>
Here is our detailed blogpost report on our this analysis - <br>
**Part 1 (how good TransWeather is able to generate clear images from foggy and rainy images) :** <a href='https://medium.com/p/c4904bcfc3ae/edit'> Transweather - An Transformer based Adverse Weather Image Enhancement Technique </a> <br>
**Part 2 ( how this resulted clear images helpful to make object detection more accurate) :** <a href='https://medium.com/p/37466a735c15/edit'> Object Detection on Transweather Generated Cleaned Images </a>
<br>

### Pipelines

Next after this experiment, we developed end to end user friendly pipelines for training, testing, inference on single image, inference on entire dataset (folder of images). These user friendly piplines are not available in the official github repository of TransWeather. To setup Transweather for our use cases we faced many issues like hard coded file paths and some dependencies. We have edited those codes and remove all hard coded things and solved all dependency issues and make this repository much more reusable. Our pipelines are developed in a way that after download or clone our repository and a one line of pip installs (may takes upto 20 mins), user can train, test, infer transweather model on their datasets. It is as simple as it sounds. Here in below the guidelines given.

## Data Format

For training or testing you must have the following - <br>
A input folder which contains the input images i.e. foggy or rainy images. You later have to pass this folder path in <br>```-input_path paste_the_path_here``` as an argument. <br>
A target folder which contains the target or ground truth images i.e. clear weather images. You later have to pass this folder path in <br>```-target_path paste_the_path_here``` as an argument.
<br>
For testing you have to create an another empty folder anywhere to store outputs or generated clear images of model. You later have to pass this folder path in ```-result_path paste_the_path_here``` as an argument. <br>

*Note : The file names of the input images in input folder and respective target images in the target folder must be same.

For inference a folder you don't need the target folder. You later have to pass the input foder path as ```-folder_path paste_the_path_here``` and a new empty folder path to store results ```-save_path paste_the_path_here```.
<br>
For inference on a single image instead of pass a input folder path, you later have to pass input image path to ```-image_path paste_the_path_here``` and a new empty folder path to store the result ```-save_path paste_the_path_here```.


## Model Format

The model format must be like - 

```
- model_path
  - model_weights
    - best
    - latest
```

Here best and latest both are the model weights file where best means the weights evaluated as best on validation dataset while training and latest means the latest or last epoch weights while training.

For all kind of tasks like training, testing, inference on both single image and a folder, you have to pass the ```model_path``` in ```-model_path paste_the_path_here``` as argument.

*Note : For training, if the model path folder is empty i.e. there is no model_weights folder then while training these all automatically create by the code itself and the model training start from the beginning. But if model_weights with best and latest weights is present there than the model starts training by initializing that weights. This way we can do checkpointing and fine tune. 
For test or inference, model_weights must be not empty and present there in model path folder.


## Initial Setup or Installation

By this following command the required libraries will installed - <br>
```
!pip install timm==0.3.2 mmcv-full==1.2.7 torch==1.7.1 torchvision==0.8.2 opencv-python==4.5.1.48 scikit-image==0.16.1
```

This is the first and mandatory step to use Transweather for all tasks i.e. train, test, inference.


## Model Training

Assuming your current working directory is this repository. If not `cd` to this repository first.
```
!python ./train.py -model_path 'paste_model_path_here' -input_path 'paste_input_folder_path_here' -target_path 'paste_target_folder_path_here' -batch_size 32 -num_epochs 100
```

## Model Testing

Assuming your current working directory is this repository. If not `cd` to this repository first.
```
!python ./test.py -model_path 'paste_model_path_here' -input_path 'paste_input_folder_path_here' -target_path 'paste_target_folder_path_here' -results_path 'paste_results_folder_path_here'
```

## Model Inference on a Single Image

Assuming your current working directory is this repository. If not `cd` to this repository first.
```
!python ./inference_image.py -model_path 'paste_model_path_here' -image_path 'paste_input_image_path_here' -save_path 'paste_save_folder_path_here'
```

## Model Inference on a Folder of Images

Assuming your current working directory is this repository. If not `cd` to this repository first.
```
!python ./inference_folder.py -model_path 'paste_model_path_here' -folder_path 'paste_input_folder_path_here' -save_path 'paste_save_folder_path_here'
```

## Demo

<p float="left">
   <img src="https://github.com/avawatz/computer-vision-interns/blob/main/Transweather/demo/fog.jpg?raw=true" width="500" />
  <img src="https://github.com/avawatz/computer-vision-interns/blob/main/Transweather/demo/rain.jpg?raw=true" width="500" />
</p>

**The top images is the ground truth, middle one is the input and buttom one is generated result.**
