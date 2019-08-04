# CWCV2019-Amur-Tiger-Re-ID

This code is mainly for the **Tiger Re-ID in the plain track** and **Tiger Re-ID in the wild track**[CVWC2019](https://cvwc2019.github.io/challenge.html) @ICCV19 Workshop.

## Getting Started
### Clone the repo:

```
git clone https://github.com/LcenArthas/CWCV2019-Amur-Tiger-Detection.git
```
#### Dependencies

Tested under python3. Ubantu16.04

- python packages
  - pytorch==0.4.1
  - torchvision>=0.2.0
  - cython
  - matplotlib
  - numpy
  - scipy
  - opencv
  - pyyaml
  - packaging
  - [pycocotools](https://github.com/cocodataset/cocoapi)  — for COCO dataset, also available from pip.
  - tensorboardX  — for logging the losses in Tensorboard
- An NVIDAI GPU and CUDA 8.0 or higher. Some operations only have gpu implementation.
