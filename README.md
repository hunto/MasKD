# Masked Distillation with Receptive Tokens (MasKD)
Official implementation of paper "Masked Distillation with Receptive Tokens" (MasKD).

By Tao Huang, Yuan Zhang, Shan You, Fei Wang, Chen Qian, Jian Cao, Chang Xu.

:fire: **MasKD: better and more general feature distillation method for dense prediction tasks (e.g., detection and segmentation).**

## Updates  

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
sh tools/mmdet/dist_train_mmdet.sh configs/distill/sfd/sota_sfd_backbone_cls_reg_cmx101_fpn-r50_coco.py 8 work_dirs/sfd/det/cm_x101_fpn_r50
```


### Results  
* Baseline settings:  

    |Student|Teacher|MasKD|Config|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |Faster RCNN-R50 (38.4)|Faster RCNN-R101 (39.8)|40.6||||
    |RetinaNet-R50 (37.4)|RetinaNet-R101 (38.9)|39.9||||
    |FCOS-R50 (38.5)|FCOS-R101 (40.8)|42.2|||

* Stronger teachers:

    |Student|Teacher|MasKD|Config|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |Faster RCNN-R50 (38.4)|Cascade Mask RCNN-X101 (45.6)|42.4|[config](mmrazor/configs/distill/maskd/cascade_mask_rcnn_x101-fpn_x50_coco.py)|||
    |RetinaNet-R50 (37.4)|RetinaNet-X101 (41.0)|40.6||||
    |RepPoints-R50 (38.6)|RepPoints-R101 (44.2)|41.4|||
### Learning masks  
You can train your own mask tokens with the code provided in `mmdetection` folder. Please check [mmdetection/README.md](mmdetection/README.md) for detailed instructions.

### Semantic segmentation  
For semantic segmentation, please see `segmentation` folder.

## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  