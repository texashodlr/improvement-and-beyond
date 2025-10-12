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
* Python:
    * matplotlib==3.10.7
    * numpy==2.3.3
    * Pillow==11.3.0
    * tqdm==4.67.1
    * torch==2.8.0+cu129
    * torchvision==0.23.0+cu129
* GPU:
    * CUDA==12.9
    * Nvidia GPU Driver Version==576.02
    * Preferably a GPU (either NVIDIA or AMD, your choice brother!)
* You'll likely get all of your [PyTorch dependencies from the site](https://pytorch.org/get-started/locally/)

## Problem
This ResNet-18 Model design focuses on building the full infrastructure necessary for training and validating the model on the Fashion-MNIST data-set.

## Motivation
This problem arises from a homework problem from my _Optimization for Deep-Learning_ and I wanted to extend the homework slightly into a full post to just recap my code and more fully understand everything that I've built.

This model aims and achieves >95% accuracy on the Fashion-MNIST data set but more importantly (to me) the code is structured into a single file which is easily comprehended to newbs.

Success critieria; model acc. >95%, code is easily digestable.

## High-level Design
As we stated above this implementation of [ResNet-18](https://arxiv.org/abs/1512.03385) _should_ be my near-direct copy and paste implementation of model as described in the paper.

### Big Idea
__What are we doing?__ We're _doing_ [residual learning](https://en.wikipedia.org/wiki/Residual_neural_network) (_"a deep learning architecture in which the layers learn residual functions with reference to the layer inputs."_). 

This means that instead of our models learning a direct mapping _H(x)_, each block learns a residual _F(x) = H(x) - x_ which then outputs _y = F(x) + x or a projected x_.

__Why are we doing this?__ We're adding this, identity path (the _'+ x'_ from the output _y_), in order to make our optimization easier, to preserve gradient flow in deep neural nets (NNs), prevent vanishing gradients and general 'degradation' (which can be understood as a rise in training error/decrease in accuracy as NN depth increases).

With the identity path means that the gradient flows __unchanged__ around the residual branch. Even if our jacobian of the residual branch is small (being early in training or due to saturations) the gradient doesn't vanish because the identity path contributes a straight shot or by-pass.

So our _main path_ computes F(x) which is the conv-> BN -> ReLU -> Conv -> BN, layer progression. The skip/identity path, provides the identity (_'+ x'_, if the shapes match) or a projection (P(x)) if the shapes are changing. We then merge everything at the output -> _y = ReLU(F(x) + x)_.

__Implementation Diagram__
Since we're using the Fashion-MNIST data set all the images are single channel 28x28 pixel images, hence we begin with an input of 1x28x28! 
```
Input (1×28×28)  # grayscale
        │
        ▼
[STEM] Conv3×3, 64, stride 1, padding 1 → BN → ReLU → Maxpool      # 64×28×28
        │
        ▼
[LAYER1] ┌─ BasicBlock(64 → 64, stride 1) ─┐              # 64×28×28
         └─ BasicBlock(64 → 64, stride 1) ─┘              # 64×28×28
        │
        ▼
[LAYER2] ┌─ BasicBlock(64 → 128, stride 2)* ─┐            # 128×14×14
         └─ BasicBlock(128 → 128, stride 1) ─┘             # 128×14×14
        │
        ▼
[LAYER3] ┌─ BasicBlock(128 → 256, stride 2)* ─┐           # 256×7×7
         └─ BasicBlock(256 → 256, stride 1) ─┘             # 256×7×7
        │
        ▼
[LAYER4] ┌─ BasicBlock(256 → 512, stride 2)* ─┐           # 512×4×4
         └─ BasicBlock(512 → 512, stride 1) ─┘             # 512×4×4
        │
        ▼
GlobalAvgPool (4×4 → 1×1)                                  # 512×1×1
        │
        ▼
FC: 512 → 10  # Fashion-MNIST classes
```
### Model Components
The below are the main _components_ that make up my ResNet-18 model.

#### Stem
This is the input frontend that turns raw images into feature maps.

For our Fashion-MNIST ResNet the flow is:
`input (x) -> 3x3 conv (stride 1, padding 1) -> BN -> ReLU -> maxpool`

This _frontend_ is converting pixels to low-level edges/textures and since we're dealing with small images we can utilize a 1-stride.

#### Layer or BasicBlock
Each layer is a _BasicBlock_ and there's four layers in the model.

For our model the flow of each layer is:
` input (x) -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> identity (x) -> ReLU`

Where the identity is the skip portion which just feeds the input after the second BN such that output of each block/layer is: `ReLU (main path + identity)`.

#### Layer Collection and Output
Since we've got four layers, we follow a channel (width) increase of: `[64, 128, 256, 512]`, and make each layer 2 blocks deep (depth) this lets our ResNet-18 model continue to learn increasingly abstract features while shrinking spatial size to keep the compute requirements reasonable.

After the four layers we leverage the 2-D Adaptive Average Pool  and terminate the model with a single fully connected linear layer to our 10 classes of fashion!

## Data Pipeline

## Model Walkthrough

## Training Setup

## Experiments and Results

## Gotchas and Debugging

## Reproducibility

## Further Developments

## Conclusion