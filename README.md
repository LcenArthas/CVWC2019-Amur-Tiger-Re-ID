CVWC2019-Amur-Tiger-Re-ID
===============

<div align="center">

<img src="data/AmurTiger/flod0/Pseudo_Mask_filp/273_c6s6_1882.body.jpg" width="900px"/>

<p> Example result of Rank-10 .</p>

</div>

:trophy:Code for 1st Place Soluition in both the **Tiger Re-ID in the plain track** and **Tiger Re-ID in the wild track**[CVWC2019](https://cvwc2019.github.io/challenge.html) @ICCV19 Workshop.

------

:running: Getting Started
-----

## :one: Clone the repo:

```
git clone https://github.com/LcenArthas/CWCV2019-Amur-Tiger-Re-ID.git
```
## :two: Dependencies

Tested under python3.6  Ubantu16.04

- python packages
  - pytorch=1.0.1
  - torchvision==0.2.1
  - pytorch-ignite=0.1.2 (Note: V0.2.0 may result in an error)
  - yacs==0.1.6
  - tensorboardx
  - h5py==2.9.0
  - imgaug==0.2.9
  - matplotlib==3.1.0
  - numpy==1.16.4
  - opencv==4.1.0.15
  - pillow==6.0.0
  - scikit-image==0.15.0
  - scipy==1.3.0
  - tensorboardx==1.6
  - tqdm==4.32.1
  - yacs==0.1.6

---------

:point_right: Section1  The Tiger Plain Re-ID:
--------

## :running: Train

### :one: Data Prearation

- [Train Dataset](https://pan.baidu.com/s/1QDRDcR4D8M3mOjX0VKd3Vw)

:small_orange_diamond: Download the train dataset and put them(atrw_reid_train, atrw_anno_reid_train) into the `{repo_root}/process_data/`. 

:small_orange_diamond: Transform the data style for the model

```
python data_process.py
```

### :two: Pre-trained weight

:small_orange_diamond: Creat a new folder named `/pretrained_model/` under the `{repo_root}/`:

```
cd data
mkdir pretrained_model
```

:small_orange_diamond: Download the pre-trained weighte and put them into the `{repo_root}/pretrained_model/`. 

- [ResNet50](https://pan.baidu.com/s/1iUsMA0k6cHYvTYbqsMSzoQ) - [ResNet101](https://pan.baidu.com/s/1g-0MRS0IYNi2yhlWXZUmAw)

- [ResNet18](https://pan.baidu.com/s/1aC-ZgxW1UVYLDUlyY6QZCQ)

- [ResNet34](https://pan.baidu.com/s/1YY0IPHF3tBqW4tQ54ZuXgA)

- [ResNet152](https://pan.baidu.com/s/1hrGAWHcqb7XzYbm-T7WARg)

- [SEResNet50](https://pan.baidu.com/s/1Jd4CA7M1IIZwnRWLTctNMw)

- [SEResNet101](https://pan.baidu.com/s/1TsOD_b9gJra-pt6RNIyuUA)

- [ResNeXt50](https://pan.baidu.com/s/1J03mTHFv9ElB4Jp7pDiQFA)

- [ResNeXt101](https://pan.baidu.com/s/1nAzGsuSTXfcPOSAZfyWsbw)

- [SEResNeXt50](https://pan.baidu.com/s/1tpWgtHd4WWiwbfU97lmg7g)

- [SEResNeXt101](https://pan.baidu.com/s/10khP6x1o5VQtQeXtFhomLA)

**And make sure the repo files as the following structure:**
 
 ```
 {repo_root}
  ├── config
  ├── configs
  ├── data
  |   ├── AmurTiger
  │   │   ├── flod0
  │   │   └── reid_test
  │   │       ├── 000000.jpg
  │   │       ├── 000004.jpg
  │   │       ├── 000005.jpg
  │   │       ├── 000006.jpg
  │   │       ├── 000008.jpg
  │   │       └── ...
  │   ├── datasets
  │   ├── samplers
  │   └── ...
  ├── engine
  ├── layers
  ├── modeling
  ├── solver
  ├── tests
  ├── trained_weight
  │   ├── resnet101-bsize_model_100.pth
  │   ├── resnet101-bsize_model_300.pth       
  │   ├── resnet101-bsize_model_301.pth
  │   ├── resnet101-bsize_model_400.pth
  │   └──...
  ├── utils
  ├── check_result.py
  ├── medo.py
  ├── medo_wide.py
  ├── test.py
  └── train.py
```     
  
:clap: Train Now!
--------

```
cd tools
python train_net_step.py
```

Eventually the trained model will be saved in `{repo_root}/tools/Outputs/`


-------

### Inference

#### Data Preparation

Creat a new folder named `/reid_test/` under the `{repo_root}/data/AmurTiger/`:

```
cd data
cd AmurTiger
mkdir reid_test
```

Put the test images in the `{repo_root}/data/AmurTiger/reid_test/`.

#### Download Pretrained Model

The trained weights are following:

 - [Trained weight](https://pan.baidu.com/s/1c9kJXWLN-g-GHSAz5EHpzQ)

Download it and create a new folder under the {repo_root} named `/trained_weight/`

```
mkdir trained_weight
```

Unzip the model.zip(there will be 8 trained weights) and put them in the `{repo_root}/trained_weight/`.

**And make sure the repo files as the following structure:**

```
  {repo_root}
  ├── config
  ├── configs
  ├── data
  |   ├── AmurTiger
  │   │   ├── flod0
  │   │   └── reid_test
  │   │       ├── 000000.jpg
  │   │       ├── 000004.jpg
  │   │       ├── 000005.jpg
  │   │       ├── 000006.jpg
  │   │       ├── 000008.jpg
  │   │       └── ...
  │   ├── datasets
  │   ├── samplers
  │   └── ...
  ├── engine
  ├── layers
  ├── modeling
  ├── solver
  ├── tests
  ├── trained_weight
  │   ├── resnet101-bsize_model_100.pth
  │   ├── resnet101-bsize_model_300.pth       
  │   ├── resnet101-bsize_model_301.pth
  │   ├── resnet101-bsize_model_400.pth
  │   └──...
  ├── utils
  ├── check_result.py
  ├── medo.py
  ├── medo_wide.py
  ├── test.py
  └── train.py
      
  ```
  
#### Inference Now!

```
python demo.py
```

This process will take about 6 minutes, just a moment, please. 

It will generate a submission in the {repo_root/}:

- **submission_plain.json** —-you can submit to the Tiger Plain Re-ID track.



## Section2  The Tiger Wild Re-ID:

### Train

TO DO...

### Inference

**In this task, it's a two-step process: Detection and Re-id**

#### Detection

Please follow this repo: [CWCV2019-Amur-Tiger-Detection](https://github.com/LcenArthas/CWCV2019-Amur-Tiger-Detection)

Note that the two repos depend on different environments(Re-ID is pytorch==1.0.1, Detection is pytorch==0.4.1)

Run scrip in above repo will generate 3 files in the {repo_root/}:

- **det_submission.json** 

- **wide_box.json** 

- **reid_test(a folder)** --it contains images that have been detected and croped.

This **wide_box.json** and **reid_test(a folder)** are what we need next.

#### Re-ID

Use this reop.

##### Data Preparation

Put **wide_box.json** and **reid_test(a folder)**(created by the detector above) under the `{repo_root}/data/AmurTiger/`.

##### Download Pretrained Model(Same as the plain re-id)

The trained weight is following:

 - [Trained weight](https://pan.baidu.com/s/1c9kJXWLN-g-GHSAz5EHpzQ)

Download it and create a new folder under the {repo_root} named `/trained_weight/`

```
mkdir trained_weight
```

Unzip the model.zip and put them into the `{repo_root}/trained_weight/`.

**And make sure the repo files as the following structure:**
 
 ```
  {repo_root}
  ├── config
  ├── configs
  ├── data
  |   ├── AmurTiger
  │   │   ├── flod0
  │   │   ├── wide_box.json
  │   │   └── reid_test
  │   │       ├── 000000.jpg
  │   │       ├── 000004.jpg
  │   │       ├── 000005.jpg
  │   │       ├── 000006.jpg
  │   │       ├── 000008.jpg
  │   │       └── ...
  │   ├── datasets
  │   ├── samplers
  │   └── ...
  ├── engine
  ├── layers
  ├── modeling
  ├── solver
  ├── tests
  ├── trained_weight
  │   ├── best_model.pth
  │   ├── resnet101-bsize_model_100.pth
  │   ├── resnet101-bsize_model_300.pth       
  │   ├── resnet101-bsize_model_301.pth
  │   ├── resnet101-bsize_model_400.pth
  │   └──...
  ├── utils
  ├── check_result.py
  ├── medo.py
  ├── medo_wide.py
  ├── test.py
  └── train.py
      
  ```
  
##### Inference Now!

```
python demo_wide.py
```

This process will take about 15 minutes, just a moment, please. 

It will generate a submission in the {repo_root/}:

- **submission_wide.json** —-you can submit to the Tiger Wide Re-ID track.

