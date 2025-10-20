---
title: "ResNet-18 from Scratch"
date: 2025-10-12
---
_Howdy!_ 

This post is an explanation of my single-file full implementation of [ResNet-18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) [Architecture](https://arxiv.org/abs/1512.03385) and features training over the [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) data set, from scratch!

[The code itself](https://github.com/texashodlr/opt4dl/blob/main/1_opt4dl_code/2_assignment/full_resnet.py): `full_resnet.py` and its purpose is to have a self-contained python script that covers all phases of model training while acting as a learning piece to deepen my own understanding of training a model.

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
Implement __ResNet-18__ from first principles, then train and evaluate it on Fashion-MNIST with a clean, single-file code path: define blocks → assemble layers → add a minimal train/test harness → plot learning curves.

## Motivation
* Turn a homework prompt into a reusable reference that I (and others) can understand at a glance.
* Keep everything in one file to make the control flow obvious for beginners—no magic imports.
* Reach strong accuracy on a small dataset while highlighting design trade-offs.

_Success criteria_: Easy-to-read code with solid accuracy on Fashion-MNIST.

## High-level Design
As we stated above this implementation of [ResNet-18](https://arxiv.org/abs/1512.03385) _should_ be my near-direct copy and paste implementation of model as described in the paper.

* BasicBlock implements residual learning with a main path (Conv-BN-ReLU-Conv-BN) and a skip path(identity or 1×1 projection).
* ResNet18 stitches blocks into 4 stages: widths [64, 128, 256, 512] with two blocks per stage, using stride-2 in the first block of stages 2–4 to downsample.
* A small-image stem (3×3, stride 1) plus an early maxpool (3×3, stride 2) prepares 28×28 inputs; a global average pool and a linear head produce 10 logits.

### Big Idea (Residual Learning)
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
[STEM] Conv3×3, 64, stride 1, padding 1 → BN → ReLU → Maxpool3x3,stride2      # 64×14×14
        │
        ▼
[LAYER1] ┌─ BasicBlock(64 → 64, stride 1) ─┐              # 64×14×14
         └─ BasicBlock(64 → 64, stride 1) ─┘              # 64×14×14
        │
        ▼
[LAYER2] ┌─ BasicBlock(64 → 128, stride 2)* ─┐            # 128×7×7
         └─ BasicBlock(128 → 128, stride 1) ─┘            # 128×7×7
        │
        ▼
[LAYER3] ┌─ BasicBlock(128 → 256, stride 2)* ─┐           # 256×4×4
         └─ BasicBlock(256 → 256, stride 1)  ─┘           # 256×4×4
        │
        ▼
[LAYER4] ┌─ BasicBlock(256 → 512, stride 2)* ─┐           # 512×2×2
         └─ BasicBlock(512 → 512, stride 1)  ─┘           # 512×2×2
        │
        ▼
GlobalAvgPool (→ 1×1)                                     # 512×1×1
        │
        ▼
FC: 512 → 10  # Fashion-MNIST classes
```
_* first block in layers 2–4 uses a 1×1 projection on the skip with the same stride as the main path. (Implemented by self.shortcut inside BasicBlock.)_

### Model Components
The below are the main _components_ that make up my ResNet-18 model.

1. _BasicBlock_
* __Main path__: Conv3×3(stride s, out=out_channels) → BN → ReLU → Conv3×3(stride 1) → BN.
* __Skip path__: if shapes change (s≠1 or channels differ), use 1×1 Conv + BN with the same stride; else identity.
* __Merge__: elementwise add, then ReLU. In code: the block stores in_channels/out_channels/stride, builds conv1/conv2, and conditionally constructs self.shortcut; in forward it computes the projection on the input when needed and adds once.
* __Why this shape logic matters__: adding tensors requires same shape; the 1×1 projection guarantees channel & spatial alignment when downsampling or widening.
2. _ResNet18_
* __Stem__: Conv3×3, stride 1 → BN → ReLU → MaxPool3×3, stride 2.
* __Stages__: _make_layer(block, out_width, num_blocks, stride) constructs each stage; the first block may downsample (stride 2), subsequent blocks use stride 1. The running self.in_channels is updated after each block to keep interfaces correct.
* __Head__: AdaptiveAvgPool2d((1,1)) → Linear(512 → 10). All of this is assembled in one class and passed img_channels=1, num_classes=10 for Fashion-MNIST.

## Data Pipeline
* __Dataset__: Fashion-MNIST train/test splits are auto-downloaded via `torchvision.datasets.FashionMNIST`.
* __Transforms__: currently ToTensor() only (pixel values → [0,1] floats). Consider adding normalization and light augmentation for stronger generalization.
* __Loaders__: shuffling train, no shuffle on test; default num_workers and pin_memory (tune these for throughput).
* _Future Improvements_: Add Normalize(mean, std) for Fashion-MNIST, modest RandomCrop(padding=2) and RandomHorizontalFlip(), and set num_workers>0, pin_memory=True when training on GPU.

## Model Walkthrough
1. __Input → Stem__: (1×28×28) → 64×28×28 via 3×3/1 conv, BN, ReLU → _downsample_ to 64×14×14 via 3×3/2 maxpool.
2. __Layer1__: two residual blocks at 64 channels, stride 1 (spatial stays 14×14).
3. __Layer2__: first block stride 2 + projection → 128×7×7, then one stride-1 block.
4. __Layer3__: stride 2 → 256×4×4, then a stride-1 block.
5. __Layer4__: stride 2 → 512×2×2, then a stride-1 block.
6. __Head__: global average pool to 512×1×1, flatten, fully connected layer to 10 classes.
_All of this produces compact, increasingly abstract features while keeping compute reasonable for small grayscale images._

## Training Setup
* __Loss__: CrossEntropyLoss.
* __Optimizer__: currently AdamW (decoupled weight decay), with configurable learning_rate and weight_decay. The script also includes commented snippets for SGD and for adding an explicit L2 penalty to the loss if you want “true L2” with Adam.
* __Epoch loop__: standard model.train() → forward → compute loss → backward() → optimizer.step(). Accuracy is computed by comparing argmax logits to labels. A mirrored test() loop runs under torch.no_grad() for evaluation.
* __Early stopping__: patience-based stopper monitors test loss; training halts after a plateau. min_delta is tied to your regularization hyperparameter for sensitivity.
* __Plots__: two PNGs—accuracy and loss vs. epoch—are saved to the working directory.

__Example CLI run__: `python full_resnet.py --seed 63 --epochs 10 --batch_size 256 --learning_rate 0.001 --weight_decay 0.0 --l2_lambda 0.00001`

## Experiments
Some variations of my model that I iterated with:
* __Optimizers and Regularization__: We alternate between _vanilla-SGD_, _Adam_ and _AdamW_ for the actual runs of our model and display some generic results below:
    * __Vanilla-SGD__: CLI `python full_resnet.py --epochs 10 --batch_size 256 --learning_rate 0.001`
    * __Adam w/out Regularization__: CLI `python full_resnet.py --epochs 10 --batch_size 256 --learning_rate 0.001 --weight_decay 0.00001` 
    * __AdamW w/out Regularization__: CLI `python full_resnet.py --epochs 10 --batch_size 256 --learning_rate 0.001 --weight_decay 0.00001`
    * __Vanilla-SGD w/Regularization__: CLI `python full_resnet.py --epochs 10 --batch_size 256 --learning_rate 0.001`
    * __Adam w/Regularization__: CLI `python full_resnet.py --epochs 10 --batch_size 256 --learning_rate 0.001 --weight_decay 0.00001` 
    * __AdamW w/Regularization__: CLI `python full_resnet.py --epochs 10 --batch_size 256 --learning_rate 0.001 --weight_decay 0.00001`
* __Additional Tweaking__: For a longer experimenting time I'd have like to explore an eta != 0.001 and vary epochs and batch sizes as well as weight decays, and lambdas across all iterations
* __Simple Result with SGD__:

![ResNet-18 Accuracy with SGD]({{ '/assets/images/2025-10-12-ResNet18-from-Scratch-images/acc.png' | relative_url}})

![ResNet-18 Loss with SGD]({{ '/assets/images/2025-10-12-ResNet18-from-Scratch-images/loss.png' | relative_url}})

## Gotchas and Debugging
Some of the issues that I initially ran into:
1. __Residual Shape Mismatches__: The skip path has to match both channels and spatial size, so projection with the 1x1 conv and BN when stride != 1 otherwise channels change. The block does this via the self.shortcut.
2. __Stride__: Apply stride at the first conv of the block and on the _skip_ projections so the adds align.
3. __Throughput__: Very noticeable difference training on an A4000 v. 4070M haha.... 

## Further Developments
Based on various different blog posts I was reading in the development of this model there's several items to improve upon:
1. Mixed Precision (AMP) would increase speeds on the my GPU(s) (like my dinky laptop 4070M)
2. Learning-rate schedulers (cosine, multistep, onecycle).
3. Improved stem for tiny/varied images, so removing the initial maxpool and switching to a 3x3 stride-1 stem.
4. Checkpointing/snapshot function.
5. Stronger/more detailed regularization via label smoothing, CutMix/MixUp, DropBlock.
6. MultiGPU instead of my single GPU check and init.

## Conclusion
I implemented ResNet-18 from scratch, trained it on Fashion-MNIST, and packaged the full training path into one readable file. The key insights: residual connections make deep optimization tractable; a small-image-aware stem helps on 28×28 inputs; and simple training hygiene (good regularization, clean eval, early stopping) goes a long way. From here, the most impactful upgrades are data normalization/augmentation, AdamW or SGD with a scheduler, and AMP for speed.