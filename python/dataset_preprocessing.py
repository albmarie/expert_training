from function_utils import initialize_datasets_dataloader
from class_utils import Distortion_mode, Distortion, Distortion_precompute
import torch
import transforms as T
from torch.utils.data import Dataset
import json
from torchvision import datasets, transforms
import time
import datetime
from PIL import Image
import os
import math
#import matplotlib.pyplot as plt #For debug purpose

################################################################################################################
##########          HYPERPARAMETERS          ###################################################################
################################################################################################################

epsilon = 1e-8

img_base_width_height = 224
bit_per_pixel_scale = 8/float((img_base_width_height)**2) #Used to convert bitstream size of a compressed image to bit per pixel rate metric

num_workers = 16

################################################################################################################
##########          MAIN          ##############################################################################
################################################################################################################

def main(distortion: Distortion, distortion_precompute_mode: int, input_size: int = 224, splits = ['train', 'val']):
    if not(distortion_precompute_mode == Distortion_precompute.SAVE_PRECOMPUTED_ENCODING_DECODING or Distortion_precompute.SAVE_PRECOMPUTED_ENCODING):
        raise Exception("Invalid distortion precomoute mode.")

    _, dataloaders, _ = initialize_datasets_dataloader(distortion, input_size, distortion_precompute_mode, num_workers, num_workers, splits)
    for phase in splits:
        print_density = math.ceil(len(dataloaders[phase])/25)

        start_time = time.time()
        if phase == 'train':
            print("\n---- TRAINING ----")
        else:
            print("\n---- VALIDATION ----")
        
        batch_i = 0
        cumulated_compressed_imgs_size, compressed_imgs_total_samples = 0, 0       
        for _, bitstreams_length in dataloaders[phase]:
            if batch_i % (5*print_density) == 0:
                print("Batch " + str(batch_i), " / ", str(len(dataloaders[phase])))
            batch_i += 1

            cumulated_compressed_imgs_size += torch.sum(bitstreams_length).item()
            compressed_imgs_total_samples += len(bitstreams_length)
            
            if batch_i % print_density == 0:
                batches_left = len(dataloaders[phase]) - (batch_i + 1)
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batch_i + 1))
                print(f"    ---- ETA {time_left}")
                
        mean_image_size = cumulated_compressed_imgs_size / (epsilon + float(compressed_imgs_total_samples))
        mean_bit_pet_pixel = bit_per_pixel_scale*mean_image_size
        
        csv_file = open("/data/_stat_compressed/stat_compressed_" + phase + "_" + str(distortion) + ".csv", "w")
        csv_file.write(str(mean_image_size) + "\n" + str(mean_bit_pet_pixel))
        csv_file.close()
                
################################################################################################################

if __name__ == "__main__":
    #Example where only val set is compressed (much faster for debug purposes)
    #main(Distortion(Distortion_mode.BPG, quality=51, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING         , 224, splits=['val']) 
    

    #We highly recommand NOT using this preprocessing mode as it makes training much slower
    """ SAVE_PRECOMPUTED_ENCODING """
    #main(Distortion(Distortion_mode.BPG, quality=41, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING         , 224)
    #main(Distortion(Distortion_mode.BPG, quality=46, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING         , 224)
    #main(Distortion(Distortion_mode.BPG, quality=51, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING         , 224)
    

    #We recommand using this preprocessing mode , even though it requires a lot of disk space. Use SAVE_PRECOMPUTED_ENCODING only if you do not have disk space for storing a copy of the uncompressed dataset 
    """ SAVE_PRECOMPUTED_ENCODING_DECODING """
    #main(Distortion(Distortion_mode.BPG, quality=41, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING_DECODING, 224)
    #main(Distortion(Distortion_mode.BPG, quality=46, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING_DECODING, 224)
    main(Distortion(Distortion_mode.BPG, quality=51, color_subsampling="420"), Distortion_precompute.SAVE_PRECOMPUTED_ENCODING_DECODING, 224)
    