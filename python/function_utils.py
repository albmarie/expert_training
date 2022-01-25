import torch
from class_utils import Distortion_mode, Distortion, Distortion_precompute
from torchvision import models
from datasets import SavePrecomputedEncodingImageNetDataset, SavePrecomputedEncodingDecodingImageNetDataset, PrecomputedOnTheFlyImageNetDataset, PrecomputedEncodingImageNetDataset, PrecomputedEncodingDecodingImageNetDataset
from torchvision import transforms
from PIL import Image
import os
from typing import List
import pandas

################################################################################################################
##########          FUNCTIONS UTILS          ###################################################################
################################################################################################################

#Run the bash command cmd. If hide_output is set to True, output of cmd command won't be printed to the standard output. 
def run_bash_cmd(cmd, hide_output=True):
    if hide_output:
        cmd = "(" + cmd + ")> /dev/null 2>&1"
    os.system(cmd)

################################################################################################################

#Return 1 if a sample is correctly classfied according to top-1, top-5, or topk metric, else 0
def accuracies_arrays(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].transpose(0, 1).int().sum(1)
            res.append(correct_k)
        
        return res

################################################################################################################

def set_model_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model

################################################################################################################

#conv_layers_fixed will be used only if requires_grad is True.
def initialize_model(model_name, use_pretrained, requires_grad=True, conv_layers_fixed=False):
    model = None
    input_size = 224

    if model_name == "resnet50":
        """ resnet50 """
        model = models.resnet50(pretrained=use_pretrained)
    elif model_name == "vgg19_bn":
        """ vgg19_bn """
        model = models.vgg19_bn(pretrained=use_pretrained)
    elif model_name == "squeezenet":
        """ squeezenet1_0 """
        model = models.squeezenet1_0(pretrained=use_pretrained)
    elif model_name == "mobilenet_v2":
        """ mobilenet_v2 """
        model = models.mobilenet_v2(pretrained=use_pretrained)
    elif model_name == "resnext101_32x8d":
        """ resnext101_32x8d """
        model = models.resnext101_32x8d(pretrained=use_pretrained)
    elif model_name == "mnasnet":
        """ mnasnet """
        model = models.mnasnet1_0(pretrained=use_pretrained)
    elif model_name == "googlenet":
        """ googlenet """
        model = models.googlenet(pretrained=use_pretrained)
    elif model_name == "inception_v3":
        """ inception_v3 """
        input_size = 299
        model = models.inception_v3(pretrained=use_pretrained)
    elif model_name == "efficientnet-b3":
        """ efficientnet-b3 """
        input_size = 300
        model = EfficientNet.from_pretrained(model_name) #If you want to use this model, add the python package "efficientnet_pytorch" to requirements.txt
    else:
        print("Invalid model name, exiting...")
        exit()
    
    if requires_grad:
        if conv_layers_fixed: #Setting only FC layers with requires_grad=True
            model = set_model_requires_grad(model, False)  
            if model_name == "resnet50":
                """ resnet50 """
                for name,param in model.named_parameters():
                    if "fc." in name:
                        param.requires_grad = True
            
            elif model_name == "vgg19_bn":
                """ vgg19_bn """
                for name,param in model.named_parameters():
                    if "classifier." in name:
                        param.requires_grad = True

            elif model_name == "squeezenet":
                """ squeezenet1_0 """
                for name,param in model.named_parameters():
                    if "classifier." in name:
                        param.requires_grad = True

            elif model_name == "mobilenet_v2":
                """ mobilenet_v2 """
                for name,param in model.named_parameters():
                    if "classifier." in name:
                        param.requires_grad = True

            elif model_name == "resnext101_32x8d":
                """ resnext101_32x8d """
                for name,param in model.named_parameters():
                    if "fc." in name:
                        param.requires_grad = True
                
            elif model_name == "mnasnet":
                """ mnasnet """
                for name,param in model.named_parameters():
                    if "classifier." in name:
                        param.requires_grad = True
            
            elif model_name == "googlenet":
                """ googlenet """
                for name,param in model.named_parameters():
                    if "fc." in name:
                        param.requires_grad = True

            elif model_name == "inception_v3":
                """ inception_v3 """
                for name,param in model.named_parameters():
                    if ".fc" in name:
                        param.requires_grad = True
            
            elif model_name == "efficientnet-b3":
                """ efficientnet-b3 """
                raise Exception("Not implemented")

            else:
                print("Invalid model name, exiting...")
                exit()
        else:
            model = set_model_requires_grad(model, True)
    else:
            model = set_model_requires_grad(model, False)

    return model, input_size

################################################################################################################

def get_distortion_mean_rate(distortion: Distortion, phase):
    file_path = "/data/_stat_compressed/stat_compressed_" + phase + "_" + str(distortion) + ".csv"
    if not os.path.exists(file_path):
        return -1, -1
    
    df = pandas.read_csv(file_path, header=None)
    mean_image_size, mean_bit_pet_pixel = df[0][0], df[0][1]
    return mean_image_size, mean_bit_pet_pixel

################################################################################################################

def get_folder_string_name(distortion: Distortion):
    
    if distortion.DISTORTION_MODE == Distortion_mode.LOSSLESS:
        return "lossless" + "/"
    
    quality_string = "_quality_" + str(distortion.quality) if distortion.QUALITY_SUPPORTED else ""
    
    color_subsampling_string = "_color_subsampling_" + distortion.color_subsampling if distortion.COLOR_SUBSAMPLING_SUPPORTED else ""
    
    folder_name = distortion.DISTORTION_STRING + quality_string + color_subsampling_string + "/"
    return folder_name

################################################################################################################

def initialize_datasets_dataloader(distortion: Distortion, input_size, distortion_precompute_mode: int, batch_size, num_workers, splits = ['train', 'val']):
    img_base_width_height = 224 #Default value used for resize and center crop while preprocessing the image for the CNN

    if input_size < img_base_width_height:
        #This is done because we do not want the square image to correspond to a small fraction of the original image
        preprocess_transform = transforms.Compose([transforms.Resize(256, interpolation=Image.BICUBIC),
                                                   transforms.CenterCrop(img_base_width_height),
                                                   transforms.Resize(input_size, interpolation=Image.BICUBIC)])
    elif input_size == img_base_width_height:
        preprocess_transform = transforms.Compose([transforms.Resize(256, interpolation=Image.BICUBIC),
                                                   transforms.CenterCrop(input_size)])
    else: #input_size >= img_base_width_height
        preprocess_transform = transforms.Compose([transforms.Resize(int(256*(input_size/img_base_width_height)), interpolation=Image.BICUBIC),
                                                   transforms.CenterCrop(int(input_size))])
    
    mean_norm, std_norm = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32), torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    postprocess_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean_norm.tolist(), std=std_norm.tolist())])
    
    undistorted_dataset_folder_string = "/data/imagenet/"
    folder_string = get_folder_string_name(distortion)
    
    if distortion_precompute_mode == Distortion_precompute.SAVE_PRECOMPUTED_ENCODING:
        print("Prepare half preprocessing computation of dataset with parameters", distortion, "...")
        datasets = {x : SavePrecomputedEncodingImageNetDataset(undistorted_dataset_folder_string, "/data/imagenet_" + folder_string, x, distortion, preprocess_transform) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=num_workers, shuffle=False, num_workers=num_workers) for x in splits}

    elif distortion_precompute_mode == Distortion_precompute.SAVE_PRECOMPUTED_ENCODING_DECODING:
        print("Prepare preprocessing computation of dataset with parameters", distortion, "...")
        datasets = {x : SavePrecomputedEncodingDecodingImageNetDataset(undistorted_dataset_folder_string, "/data/imagenet_" + folder_string, x, distortion, preprocess_transform) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=num_workers, shuffle=False, num_workers=num_workers) for x in splits}

    elif distortion_precompute_mode == Distortion_precompute.PRECOMPUTED_ON_THE_FLY:
        print("Dataset with parameters", distortion, "will be computed on-the-fly...")
        datasets = {x : PrecomputedOnTheFlyImageNetDataset(undistorted_dataset_folder_string, x, distortion, preprocess_transform, postprocess_transform) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in splits}

    elif distortion_precompute_mode == Distortion_precompute.PRECOMPUTED_ENCODING:
        print("Loading half precomputed dataset with parameters", distortion, "...")
        datasets = {x : PrecomputedEncodingImageNetDataset(undistorted_dataset_folder_string, "/data/imagenet_" + folder_string, x, distortion, preprocess_transform, postprocess_transform, input_size) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in splits}

    elif distortion_precompute_mode == Distortion_precompute.PRECOMPUTED_ENCODING_DECODING:
        print("Loading precomputed dataset with parameters", distortion, "...")
        datasets = {x : PrecomputedEncodingDecodingImageNetDataset(undistorted_dataset_folder_string, "/data/imagenet_" + folder_string, x, distortion, preprocess_transform, postprocess_transform) for x in splits}
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True) for x in splits}

    else:
        raise Exception("Unknown distortion precompute mode value " + str(distortion_precompute_mode)) 

    return datasets, dataloaders, folder_string

################################################################################################################
################################################################################################################
################################################################################################################
