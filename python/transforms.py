
from PIL import Image
import numpy as np
from torchvision import transforms
import io
from typing import List
import random
import string
import os
import function_utils as utils
    
####################################################################################################
##########          TRANSFORMATIONS          #######################################################
####################################################################################################

"""
Do not perform anything.
Used for "Lossless" compression mode (since no artifacts are added to provided sample image, we simply return it as it is).
"""
class Identity(object):
    def __init__(self):
        self.len = -1
        self.codec = "LOSSLESS"
        
    def __call__(self, sample):
        return sample

####################################################################################################

"""
Perform a JPEG compression with the provided quality parameter with a probability p.

Require a RGB PIL image, return a RGB PIL image.
"""
class JPEG(object):
    def __init__(self, quality: int, color_subsampling = "420", optimize = False, p: float = 1.0):
        self.quality = quality
        self.color_subsampling = color_subsampling
        self.optimize = optimize
        self.bitstream_format=".jpg"
        self.p = p
        self.len = -1
        self.codec = "JPEG"

    def encoder(self, sample, image_path):
        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(image_path + self.bitstream_format, self.codec, quality=self.quality, optimize=self.optimize)
        else:
            sample.save(image_path + self.bitstream_format, self.codec, quality=self.quality, subsampling=self.color_subsampling[0] + ":" + self.color_subsampling[1] + ":" + self.color_subsampling[2], optimize=self.optimize)

    def decoder(self, image_path):
        return Image.open(image_path + self.bitstream_format).convert('RGB')
    
    def free_memory(self, image_path, keep_bitstream):
        if not keep_bitstream:
            rm_cmd = "rm " + image_path + self.bitstream_format
            utils.run_bash_cmd(rm_cmd)

    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample

        buffer = io.BytesIO()
        if self.color_subsampling == "400": #Greyscale
            sample.convert('L').save(buffer, self.codec, quality=self.quality, optimize=self.optimize)
        else:
            sample.save(buffer, self.codec, quality=self.quality, subsampling=self.color_subsampling[0] + ":" + self.color_subsampling[1] + ":" + self.color_subsampling[2], optimize=self.optimize)
        self.len = buffer.getbuffer().nbytes
        return Image.open(buffer).convert('RGB')

####################################################################################################

"""
Perform a JPEG2000 compression with the provided quality_layers parameter with a probability p.

Require a RGB PIL image, return a RGB PIL image.
"""
class JPEG2000(object):
    def __init__(self, quality_layers: List[int], p: float = 1.0):
        self.quality_layers = quality_layers
        self.bitstream_format=".jp2"
        self.p = p
        self.len = -1
        self.codec = "JPEG2000"

    def encoder(self, sample, image_path):
        sample.save(image_path + self.bitstream_format, self.codec, quality_layers=self.quality_layers)

    def decoder(self, image_path):
        return Image.open(image_path + self.bitstream_format).convert('RGB')
    
    def free_memory(self, image_path, keep_bitstream):
        if not keep_bitstream:
            rm_cmd = "rm " + image_path + self.bitstream_format
            utils.run_bash_cmd(rm_cmd)
    
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample
        
        buffer = io.BytesIO()
        sample.save(buffer, self.codec, quality_layers=self.quality_layers)
        self.len = buffer.getbuffer().nbytes
        return Image.open(buffer)

####################################################################################################

"""
Perform a BPG compression (https://bellard.org/bpg/) with the provided qP parameter with a probability p.

Require a RGB PIL image, return a RGB PIL image.
"""
class BPG(object):
    def __init__(self, qP: int, color_subsampling = "420", p: float = 1.0):
        self.qP = qP
        self.color_subsampling = color_subsampling
        self.in_format=".png"
        self.bitstream_format=".bpg"
        self.out_format=".ppm"
        self.p = p
        self.len = -1
        self.codec = "BPG"
    
    def encoder(self, sample, image_path):
        sample.save(image_path + self.in_format)

        enc_cmd  = "bpgenc "
        enc_cmd += image_path + self.in_format
        enc_cmd += " -o " + image_path + self.bitstream_format
        enc_cmd += " -m 9 -b 8 -c ycbcr"
        enc_cmd += " -q " + str(self.qP)
        enc_cmd += " -f " + self.color_subsampling + " &&"
        enc_cmd += " rm " + image_path + self.in_format
        utils.run_bash_cmd(enc_cmd)
        self.len = os.stat(image_path + self.bitstream_format).st_size

    def decoder(self, image_path):
        dec_cmd  = "bpgdec "
        dec_cmd += image_path + self.bitstream_format
        dec_cmd += " -o " + image_path + self.out_format
        utils.run_bash_cmd(dec_cmd)
        return Image.open(image_path + self.out_format).convert('RGB')
    
    def free_memory(self, image_path, keep_bitstream):
        rm_cmd  = "rm " + image_path + self.out_format
        rm_cmd += "" if keep_bitstream else (" && rm " + image_path + self.bitstream_format)
        utils.run_bash_cmd(rm_cmd)
    
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample

        image_path = "/mnt/ram_partition/_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=16))
        self.encoder(sample, image_path)
        compressed_image = self.decoder(image_path)
        self.free_memory(image_path, keep_bitstream=False)

        return compressed_image

####################################################################################################

"""
Resize the provided image.
A factor of n means that the final image will have its width and height multiplied by n. Therefore, aspect ratio will always be preserved.
The paramter n can be inferior to 1. It will result in smaller images.

Require a PIL image, return a PIL image.
"""
class Resize(object):
    def __init__(self, factor: float, interpolation = Image.BICUBIC, p: float = 1.0):
        self.p = p
        self.factor = factor
        self.interpolation = interpolation
        self.len = -1
        
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample
            
        self.len = -1
        sample_width, sample_height = sample.size
        sample_width, sample_height = round(sample_width*self.factor), round(sample_height*self.factor)
        return sample.resize((sample_width, sample_height), resample=self.interpolation)

####################################################################################################

"""
Add an additive white gaussian noise to the provided sample with a probability p.

Require a numpy array, return a numpy array.
"""
class AdditiveWhiteGaussianNoise(object):
    def __init__(self, uniform_gaussian_noise_std, p: float = 1.0):
        self.uniform_gaussian_noise_std = uniform_gaussian_noise_std
        self.p = p
        self.len = -1
        
    def __call__(self, sample):
        rand = np.random.uniform()
        perform_transform = rand <= self.p
        if not perform_transform:
            self.len = 0
            return sample

        self.len = -1
        noise = np.random.normal(0, 256*self.uniform_gaussian_noise_std, np.array(sample).shape)
        return Image.fromarray(np.clip(np.array(sample).astype(np.float64) + noise, 0, 255).astype(np.uint8))

####################################################################################################
####################################################################################################
####################################################################################################
