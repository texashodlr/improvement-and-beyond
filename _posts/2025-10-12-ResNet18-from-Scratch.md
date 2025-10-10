---
title: "ResNet-18 from Scratch"
date: 2025-10-12
---
# ResNet-18 from Scratch
_Howdy!_ 

This post is an explanation of my single-file full implementation of [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) [Architecture](https://arxiv.org/abs/1512.03385) and features training over the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) data set, from scratch!

The code itself: `full_resnet.py` and its purpose is to have a self-contained python script that covers all phases of model training while acting as a learning piece to deepen my own understanding of training a model.

## Prerequisites
To implement this code you'll need:
* matplotlib==3.10.7
* numpy==2.3.3
* Pillow==11.3.0
* tqdm==4.67.1
* torch==2.8.0+cu129
* torchvision==0.23.0+cu129
* CUDA==12.9
* Driver Version==576.02
* Preferably a GPU (either NVIDIA or AMD, your choice brother!)
* Probably can just download all of your [PyTorch from the site](https://pytorch.org/get-started/locally/)

## Problem

## Motivation