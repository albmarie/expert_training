# Expert Training

This repo is the implementation of the paper titled "Expert Training: Enhancing AI Resilience to Image Coding Artifacts", published at Electronic Imaging (EI) in 2022.

## Installation

### Requirements
With these minimal requirements, all the scripts in this repo can be executed.

- Docker 
- Python 3
- [Argparse](https://pypi.org/project/argparse/) package
- Singularity (optional, allow to run scripts without sudo privileges)

### Singularity (optional)
To avoid using Docker to run our scripts, which requires sudo privileges, singularity can be used instead.
To use singularity, you need to build the singularity container `expert_training_container.sif` out of docker container `expert_training_container:latest` first.

To do so, cd to the root of this repo and build the docker image using:
```
sudo docker build -t expert_training_container:latest --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```
Note that this step is not necessary if you already built the docker image by running one of *run_\*\*.py* scripts.

Afterwards, build the singularity container out of the docker container using:
```
sudo singularity build singularity/expert_training_container.sif docker-daemon://expert_training_container:latest
```

## Get Started
3 scripts are provided to reproduce main results of the paper, while keeping the code as easy as possible. These scripts are:

- [run_expert_training.py](run_expert_training.py) / [expert_training.py](python/expert_training.py)
- [run_dataset_preprocessing.py](run_dataset_preprocessing.py) / [dataset_preprocessing.py](dataset_preprocessing.py)
- [run_stability_training.py](run_stability_training.py) / [stability_training.py](stability_training.py)

Each *run_\*\*.py* script consists of the following steps:
- Build a docker image (expert_training_container:latest), using the provided [Dockerfile](Dockerfile).
- Handle provided arguments in the command line with argparse
- Run the corresponding *\*\*.py* script on the docker/singularity container.

### Folder structure 
Each script expect a certain folder structure to work.

```
/path/to/my/folder/containing/imagenet/dataset/folder/
├── imagenet/
│   ├── train/
│   │   ├── n01440764/
│   │   ├── n01443537/
│   │   ├── ...
│   ├── val/
│   │   ├── n01440764/
│   │   ├── n01443537/
│   │   ├── ...
│   ├── ILSVRC2012_devkit_t12.tar.gz
│   ├── meta.bin
├── _preprocess_undistorted_output/
│   ├── ...
├── _stat_compressed/
│   ├── ...
```

- `imagenet/` folder refer to the ILSVRC12 dataset, containing all images of the 1000 classes (~1.3M).
- `_preprocess_undistorted_output/` is a folder used by the [expert_training.py](expert_training.py) script to precompute model f output on undistorted images when using the expert training procedure. For more details, please refer to the paper and the code. At first, simply create an empty folder.
- `_stat_compressed/` is a folder used by the [dataset_preprocessing.py](dataset_preprocessing.py) script to store rate information about preprocessed datasets. At first, simply create an empty folder.

### 1) expert_training.py
This script is used to train an ImageNet classifier using fine-tuning or expert training. To run this script, simply use the following command line:

```
python3 run_expert_training.py --gpu \
    -d /path/to/my/folder/containing/imagenet/dataset/folder/ \
    -w /path/where/model/checkpoints/will/be/saved/ \
    -c /path/where/training/informations/will/be/saved/
```

For more information, run `python3 run_expert_training.py -h`.
By default, a training using the expert training procedure will be run using JPEG compression (quality 10) with ResNet50. To change the training procedure, the distortion used or classifier, please refer to the end of the [expert_training.py](expert_training.py) script to comment/uncomment any of the other provided configurations.

Please note that, for some of the considered distortion in the paper (when the used codec is BPG), you will have to perform a preprocessing step using the [dataset_preprocessing.py](dataset_preprocessing.py) script first.

### 2) dataset_preprocessing.py
This script is used to preprocess a distortion that will be used by the [expert_training.py](expert_training.py) script afterwards. To run this script, simply use the following command line:

```
python3 run_dataset_preprocessing.py \
    -d /path/to/my/folder/containing/imagenet/dataset/folder/ \
```

For more information, run `python3 run_dataset_preprocessing.py -h`.
By default, a preprocessing will be run on the whole ImageNet dataset, using BPG codec, with quality 51.

What we call a preprocessing can be:
1. **Precomputing the encoding** of a given codec at a given quality (e.g. BPG, quality 41) of all images in the ImageNet dataset (~1.3M) by saving images bitstreams on the disk
2. **Precomputing the encoding AND the decoding** of a given codec at a given quality, by saving losslessly images that were compressed/decompressed on the disk (using .ppm image format)

We recommend using the second preprocessing every time, unless you are highly limited by disk space. For more information, please refer to comments in [datasets.py](datasets.py) and at the end of [expert_training.py](expert_training.py).

### 3) stability_training.py
This script is used to train an ImageNet classifier using stability training. To run this script, simply use the following command line:

```
python3 run_stability_training.py --gpu \
    -d /path/to/my/folder/containing/imagenet/dataset/folder/ \
    -w /path/where/model/checkpoints/will/be/saved/ \
    -c /path/where/training/informations/will/be/saved/
```

By default, a training on ResNet50 is run. The selected hyperparameters correspond to the ones in our paper, on which we obtained the best results among the considered ones.
