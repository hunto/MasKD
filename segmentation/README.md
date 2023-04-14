This directory contains the segmentation code of MasKD (Masked Knowledge Distillation with Receptive Tokens).

## Preparation

### Dataset  
Put the Cityscapes dataset into `./data/cityscapes` folder.

### Pretrained checkpoints  
Download the required checkpoints into `./ckpts` folder.
Backbones pretrained on ImageNet:
* [resnet101-imagenet.pth](https://drive.google.com/file/d/1V8-E4wm2VMsfnNiczSIDoSM7JJBMARkP/view?usp=sharing) 
* [resnet18-imagenet.pth](https://drive.google.com/file/d/1_i0n3ZePtQuh66uQIftiSwN7QAUlFb8_/view?usp=sharing) 
* [mobilenetv2-imagenet.pth](https://drive.google.com/file/d/12EDZjDSCuIpxPv-dkk1vrxA7ka0b0Yjv/view?usp=sharing) 

Teacher backbones:
* [deeplabv3_resnet101_citys_best_model.pth](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing)

## Performance on Cityscapes

Student models are trained on 8 * NVIDIA Tesla V100 GPUs.

`*: The backbone parameters are random initialized.`
|Role|Network|Method|val mIoU|test mIoU|train script|log|ckpt|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Teacher|DeepLabV3-ResNet101|-|78.07|77.46|[sh](./train_scripts/train_baseline/deeplabv3_res101.sh)|-|[Google Drive](https://drive.google.com/file/d/1zUdhYPYCDCclWU3Wo7GbbTlM8ibQ_UC1/view?usp=sharing)|
|Student|DeepLabV3-ResNet18|MasKD|77.00|75.59|[sh](./train_scripts/train_maskd/deeplabv3_res18.sh)|||
|Student|DeepLabV3-ResNet18*|MasKD|73.95|73.74|[sh](./train_scripts/train_maskd/deeplabv3_res18_rinit.sh)|||
|Student|DeepLabV3-MBV2|MasKD|75.26|74.23|[sh](./train_scripts/train_maskd/deeplabv3_mbv2.sh)|||
|Student|PSPNet-ResNet18|MasKD|75.34|74.61|[sh](./train_scripts/train_maskd/pspnet_res18.sh)|||

## Evaluate pre-trained models on Cityscapes val and test sets

### Evaluate the pre-trained models on val set
```
python -m torch.distributed.launch --nproc_per_node=8 eval.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store log files] \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth
```

### Generate the resulting images on test set

You can use [test_deeplabv3_mbv2.sh](./train_scripts/train_maskd/test_deeplabv3_mbv2.sh), [test_deeplabv3_res18.sh](./train_scripts/train_maskd/test_deeplabv3_res18.sh), and [test_pspnet_res18.sh](./train_scripts/train_maskd/test_pspnet_res18.sh) to test the student models, or use the script manually as follows:
```
python -m torch.distributed.launch --nproc_per_node=4 test.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --data [your dataset path]/cityscapes/ \
    --save-dir [your directory path to store resulting images] \
    --gpu-id 0,1,2,3 \
    --save-pred \
    --pretrained [your checkpoint path]/deeplabv3_resnet101_citys_best_model.pth
```
You can submit the resulting images to the [Cityscapes test server](https://www.cityscapes-dataset.com/submit/).

## Train Your Own Mask Module

Our pretrained mask module on deeplabv3-r101 is in [[link]](https://github.com/hunto/MasKD/releases/download/v0.0.1/maskd_seg_deeplabv3_resnet101_citys.pth) or `work_dirs/dv3-r101/deeplabv3_resnet101_citys_best_model.pth`.

You can train your own mask module with the following script:
```shell
sh train_scripts/train_mask_module/deeplabv3_res101.sh
```

Here is an example code to visualize the learned masks:
```shell
sh train_scripts/train_mask_module/vis_deeplabv3_res101.sh
```


## Acknowledgement

The code is mostly based on the code in [CIRKD](https://github.com/winycg/CIRKD.git).

