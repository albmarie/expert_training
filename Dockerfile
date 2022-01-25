FROM nvcr.io/nvidia/cuda:10.1-cudnn8-runtime-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
	apt install -y python3 python3-pip python3-tk libsm6 libxext6 libxrender-dev libgl1-mesa-glx imagemagick libgtk2.0-dev libopenjp2-7-dev

# Build libbpg from source
RUN cd / && \
	apt update && \
	apt install -y wget cmake libsdl1.2-dev libsdl-image1.2-dev yasm && \
	wget https://bellard.org/bpg/libbpg-0.9.8.tar.gz && \
	tar -xvf libbpg-0.9.8.tar.gz libbpg-0.9.8/ && \
	rm libbpg-0.9.8.tar.gz && \
	cd libbpg-0.9.8/ && \
	make -j 8 && \
	make install

ADD requirements.txt /
RUN cd / && \
	mkdir python && \
	pip3 install -r requirements.txt

ARG USER_ID
ARG GROUP_ID
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user