# Masked Distillation with Receptive Tokens (MasKD)
Official implementation of paper "[Masked Distillation with Receptive Tokens](https://arxiv.org/abs/2205.14589)" (MasKD), ICLR 2023.

By Tao Huang*, Yuan Zhang*, Shan You, Fei Wang, Chen Qian, Jian Cao, Chang Xu.

:fire: **MasKD: better and more general feature distillation method for dense prediction tasks (e.g., detection and segmentation).**

<p align='center'>
<img src='./assests/mask.png' alt='mask' width='1000px'>
</p>

## Updates  
### March 04, 2023  
Configs for KD are available in `mmrazor` folders and student checkpoints are provided.
### May 30, 2022  
Code for mask learning and KD is available in `mmdetection` and `mmrazor` folders.

## Reproducing our results

### Train students with pretrained masks  
We provide the learned pretrained mask tokens in our experiments at [release](https://github.com/hunto/MasKD/releases/tag/v0.0.1).

This repo uses [MMRazor](https://github.com/open-mmlab/mmrazor) as the knowledge distillation toolkit. For environment setup, please see [mmrazor/README.md](mmrazor/README.md).

**Train student:**  
```shell
cd mmrazor
sh tools/mmdet/dist_train_mmdet.sh ${CONFIG} 8 ${WORK_DIR}
```

Example for reproducing our `cascade_mask_rcnn_x101-fpn_r50` result:
```shell
sh tools/mmdet/dist_train_mmdet.sh configs/distill/maskd/cascade_mask_rcnn_x101-fpn_x50_coco.py 8 work_dirs/maskd_cmr_x101-fpn_x50
```


### Results  
* Baseline settings:  

    |Student|Teacher|MasKD|Config|Log|CheckPoint|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |Faster RCNN-R50 (38.4)|Faster RCNN-R101 (39.8)|41.0|[config](mmrazor/configs/distill/maskd/fpn_r101-fpn_r50_coco.py)|[log](https://github.com/Gumpest/MasKD/releases/download/v0.0.3/fpn_r101-fpn_r50_coco.json)|[GoogleDrive](https://drive.google.com/file/d/1FdOOKLGq8q3A4iYF89khyCaXoeHTnORW/view?usp=share_link)|
    |RetinaNet-R50 (37.4)|RetinaNet-R101 (38.9)|39.9|[config](mmrazor/configs/distill/maskd/retinanet_r101-retinanet_r50_coco.py)|[log](https://github.com/Gumpest/MasKD/releases/download/v0.0.3/retinanet_r101-retinanet_r50_coco.json)|[GoogleDrive](https://drive.google.com/file/d/15U2PSfUFOZVPFL3GCEpoeJ-zEfZTAgqg/view?usp=share_link)|
    |FCOS-R50 (38.5)|FCOS-R101 (40.8)|42.9|[config](mmrazor/configs/distill/maskd/fcos_r101-fcos_r50_coco.py)|[log](https://github.com/Gumpest/MasKD/releases/download/v0.0.3/fcos_r101-fcos_r50_coco.json)|[GoogleDrive](https://drive.google.com/file/d/1K-mqpWG-axIKzHX5kI79kEnJtClc-feh/view?usp=share_link)|

* Stronger teachers:

    |Student|Teacher|MasKD|Config|Log|CheckPoint|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |Faster RCNN-R50 (38.4)|Cascade Mask RCNN-X101 (45.6)|42.7|[config](mmrazor/configs/distill/maskd/cascade_mask_rcnn_x101-fpn_x50_coco.py)|[log](https://github.com/Gumpest/MasKD/releases/download/v0.0.3/cascade_mask_rcnn_x101-fpn_r50_coco.json)|[GoogleDrive](https://drive.google.com/file/d/1EXC9cGwDZ9UuaDaAsngwJHWPoQ8A49HO/view?usp=share_link)|
    |RetinaNet-R50 (37.4)|RetinaNet-X101 (41.0)|41.0|[config](mmrazor/configs/distill/maskd/retinanet_x101-retinanet_r50_coco.py)|[log](https://github.com/Gumpest/MasKD/releases/download/v0.0.3/retinanet_x101-retinanet_r50_coco.json)|[GoogleDrive](https://drive.google.com/file/d/1bioVmJpTuEvInpKHaGUvwWDxSop3eQ9I/view?usp=share_link)|
    |RepPoints-R50 (38.6)|RepPoints-R101 (44.2)|42.5|[config](mmrazor/configs/distill/maskd/reppoints_x101-reppoints-r50_coco.py)|[log](https://github.com/Gumpest/MasKD/releases/download/v0.0.3/reppoints_x101-reppoints-r50_coco.json)|[GoogleDrive](https://drive.google.com/file/d/1QDkE4_tlWq2Cw2F3aUtqCTm4sMkIxw2H/view?usp=sharing)|
### Learning masks  
You can train your own mask tokens with the code provided in `mmdetection` folder. Please check [mmdetection/README.md](mmdetection/README.md) for detailed instructions.

### Semantic segmentation  
For semantic segmentation, please see `segmentation` folder.

## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  
```
@inproceedings{
huang2023masked,
title={Masked Distillation with Receptive Tokens},
author={Tao Huang and Yuan Zhang and Shan You and Fei Wang and Chen Qian and Jian Cao and Chang Xu},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=mWRngkvIki3}}
```