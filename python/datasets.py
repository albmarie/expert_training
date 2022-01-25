from torchvision import datasets
from torch.utils.data import Dataset
from torchvision import datasets
from class_utils import Distortion
from PIL import Image
import transforms as T

import os
import re

################################################################################################################
##########          DATASETS          ##########################################################################
################################################################################################################

"""
Dataset to get add Additive Whitre Gaussian Noise (AWGN) to an image, with a standard deviation of AWGN_std.

Used in stability_training.py, a script that implement the paper called "Improving the robustness of deep neural networks via stability training".
"""
class AWGNImageNetDataset(datasets.ImageNet):
    def __init__(self, undistorted_root: str, split: str, AWGN_std, preprocess_transform, postprocess_transform) -> None:
        self.awgn_transform = T.AdditiveWhiteGaussianNoise(AWGN_std, p=1.0)
        self.postprocess_transform = postprocess_transform
        
        super(AWGNImageNetDataset, self).__init__(undistorted_root, split=split, transform=preprocess_transform)
    
    def __getitem__(self, idx):
        sample, label = datasets.ImageNet.__getitem__(self, idx)
        undistorted_sample = self.postprocess_transform(sample)
        distorted_sample   = self.postprocess_transform(self.awgn_transform(sample))

        return idx, undistorted_sample, distorted_sample, label

################################################################################################################

"""
Dataset to save bitstream on disk, so that encoding step is precomputed at training time.
Only the decoding will have to be performed on-the-fly at training time (low complexity).
A "low" storage space required ("low" -> depends on the codec/quality/etc. selected)

This dataset is not meant to be used for training/evaluating, but only for precomputing purpose!
A dataset class is used here only to use multiple threads (instead of loading 1 image like a normal Dataset class, __getitem__ compress and save the bitstream of the compressed image on disk) 
"""
class SavePrecomputedEncodingImageNetDataset(Dataset):
    def __init__(self, undistorted_root: str, distorted_root: str, split: str, distortion: Distortion, preprocess_transform) -> None:
        self.undistorted_root = undistorted_root
        self.distorted_root = distorted_root
        
        self.compress_transform = distortion.get_compress_transform()
        
        self.extension_discriminator = re.compile(r'\.JPEG')
        self.dataset = datasets.ImageNet(undistorted_root, split=split, transform=preprocess_transform)
        self.len = len(self.dataset)
 
    def __len__(self):
        return self.len       

    def __getitem__(self, idx):
        preprocessed_sample, label = self.dataset[idx]
        
        distorted_sample_filepath = self.dataset.imgs[idx][0]
        distorted_sample_filepath = distorted_sample_filepath.replace(self.undistorted_root, self.distorted_root, 1)
        os.makedirs(os.path.dirname(distorted_sample_filepath), exist_ok=True)

        self.compress_transform.encoder(preprocessed_sample, distorted_sample_filepath.replace(self.extension_discriminator.search(distorted_sample_filepath).group(0), ""))
        bitstream_length = self.compress_transform.len
        self.compress_transform.free_memory(distorted_sample_filepath.replace(self.extension_discriminator.search(distorted_sample_filepath).group(0), ""), keep_bitstream=True)

        return label, bitstream_length

################################################################################################################

"""
Dataset to save precomputed images on disk (saved losslessly), so that encoding and decoding step is precomputed at training time.
Neither the encoding nor the decoding step have to be performed on-the-fly at training time (lowest complexity).
High storage space required (saved images are not compressed for fast decoding). To save the 1.332.167 images from ImageNet dataset, (224x224, 8bits, RGB), a total of 1332167*3*224**2 = 200.528.434.176 Bytes

This dataset is not meant to be used for training/evaluating, but only for precomputing purpose!
A dataset class is used here only to use multiple threads (instead of loading 1 image like a normal Dataset class, __getitem__ downsample, compress, decompress and save a lossless version of the decoded image on disk).
"""
class SavePrecomputedEncodingDecodingImageNetDataset(Dataset):
    def __init__(self, undistorted_root: str, distorted_root: str, split: str, distortion: Distortion, preprocess_transform) -> None:
        self.undistorted_root = undistorted_root
        self.distorted_root = distorted_root
        
        self.compress_transform = distortion.get_compress_transform()

        self.extension_discriminator = re.compile(r'\.JPEG')
        self.dataset = datasets.ImageNet(undistorted_root, split=split, transform=preprocess_transform)
        self.len = len(self.dataset)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        preprocessed_sample, label = self.dataset[idx]
        
        distorted_sample_filepath = self.dataset.imgs[idx][0]
        distorted_sample_filepath = distorted_sample_filepath.replace(self.undistorted_root, self.distorted_root, 1)
        os.makedirs(os.path.dirname(distorted_sample_filepath), exist_ok=True)

        compressed_sample = self.compress_transform(preprocessed_sample)
        compressed_sample.save(distorted_sample_filepath.replace(self.extension_discriminator.search(distorted_sample_filepath).group(0), '.ppm')) #Saved losslessly on disk

        return label, self.compress_transform.len

################################################################################################################

"""
Dataset to compute on-the-fly compression -> decompression
Highest complexity, "no storage" space required ("no storage" -> some codecs use disk space for temporary files)
"""
class PrecomputedOnTheFlyImageNetDataset(datasets.ImageNet):
    def __init__(self, undistorted_root: str, split: str, distortion: Distortion, preprocess_transform, postprocess_transform) -> None:
        self.compress_transform = distortion.get_compress_transform()
        self.postprocess_transform = postprocess_transform
        
        super(PrecomputedOnTheFlyImageNetDataset, self).__init__(undistorted_root, split=split, transform=preprocess_transform)
    
    def __getitem__(self, idx):
        sample, label = datasets.ImageNet.__getitem__(self, idx)
        undistorted_sample = self.postprocess_transform(sample)
        distorted_sample   = self.postprocess_transform(self.compress_transform(sample))

        return idx, undistorted_sample, distorted_sample, label, self.compress_transform.len

################################################################################################################

"""
Dataset to load bitstream that were compressed. Therefore, this dataset perform decoding on-the-fly.
Complexity and storage required are "low" ("low" -> depends on the codec/quality/etc. selected)
"""
class PrecomputedEncodingImageNetDataset(datasets.ImageNet):
    def __init__(self, undistorted_root: str, distorted_root: str, split: str, distortion: Distortion, preprocess_transform, postprocess_transform, input_size) -> None:
        self.undistorted_root = undistorted_root
        self.distorted_root = distorted_root
        
        self.distortion = distortion
        self.compress_transform = self.distortion.get_compress_transform()
        self.preprocess_transform = preprocess_transform
        self.postprocess_transform = postprocess_transform
        
        self.input_size = input_size
        self.extension_discriminator = re.compile(r"\." + self.compress_transform.bitstream_format[1:])
        super(PrecomputedEncodingImageNetDataset, self).__init__(self.distorted_root, split=split, is_valid_file=self.is_valid_file, loader=self.bitstream_loader, transform=self.postprocess_transform)
    
    def bitstream_loader(self, path: str) -> Image.Image:
        path_without_extension = path.replace(self.extension_discriminator.search(path).group(0), "")
        img = self.compress_transform.decoder(path_without_extension)
        return img
        
    def base_loader(self, path: str) -> Image.Image:
        #open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def is_valid_file(self, path: str):
        return self.extension_discriminator.search(path) is not None

    def __getitem__(self, idx):
        distorted_sample, label = datasets.ImageNet.__getitem__(self, idx)

        undistorted_sample_path = self.imgs[idx][0]
        undistorted_sample_path = undistorted_sample_path.replace(self.distorted_root, self.undistorted_root)
        undistorted_sample_path = undistorted_sample_path.replace(self.extension_discriminator.search(undistorted_sample_path).group(0), ".JPEG")
        undistorted_sample = self.postprocess_transform(self.preprocess_transform(self.base_loader(undistorted_sample_path)))

        return idx, undistorted_sample, distorted_sample, label, os.stat(self.imgs[idx][0]).st_size

################################################################################################################

"""
Dataset to load image saved in lossless format, that were compressed and decompressed before.
Lowest complexity, highest storage space required.
"""
class PrecomputedEncodingDecodingImageNetDataset(datasets.ImageNet):
    def __init__(self, undistorted_root: str, distorted_root: str, split: str, distortion: Distortion, preprocess_transform, postprocess_transform) -> None:
        self.undistorted_root = undistorted_root
        self.distorted_root = distorted_root
        
        self.preprocess_transform = preprocess_transform
        self.postprocess_transform = postprocess_transform
        
        self.extension_discriminator = re.compile(r'\.ppm')
        super(PrecomputedEncodingDecodingImageNetDataset, self).__init__(distorted_root, split=split, is_valid_file=self.is_valid_file, loader=self.base_loader, transform=self.postprocess_transform)
    
    def is_valid_file(self, path: str):
        return self.extension_discriminator.search(path) is not None
    
    def base_loader(self, path: str) -> Image.Image:
        #open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, idx):
        distorted_sample, label = datasets.ImageNet.__getitem__(self, idx)

        undistorted_sample_path = self.imgs[idx][0]
        undistorted_sample_path = undistorted_sample_path.replace(self.distorted_root, self.undistorted_root)
        undistorted_sample_path = undistorted_sample_path.replace(self.extension_discriminator.search(undistorted_sample_path).group(0), '.JPEG')
        undistorted_sample = self.postprocess_transform(self.preprocess_transform(self.base_loader(undistorted_sample_path)))

        return idx, undistorted_sample, distorted_sample, label, os.stat(self.imgs[idx][0]).st_size

################################################################################################################