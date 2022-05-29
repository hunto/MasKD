# Masked Distillation with Receptive Tokens (MasKD)
Official implementation of paper "Masked Distillation with Receptive Tokens" (MasKD).

By Tao Huang, Yuan Zhang, Shan You, Fei Wang, Chen Qian, Jian Cao, Chang Xu.

:fire: **MasKD: better and more general feature distillation method for dense prediction tasks (e.g., detection and segmentation).**

## Updates  
Code for KD based on MMRazor is coming soon.
### May 29, 2022  
Code for mask learning is available in `mmdetection` folder.

## Reproducing our results

### Train students with pretrained masks  
We provide the learned pretrained mask tokens in our experiments at [].

* Baseline settings:  

    |Student|Teacher|MasKD|Config|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |Faster RCNN-R50 (38.4)|Faster RCNN-R101 (39.8)|40.6||||
    |RetinaNet-R50 (37.4)|RetinaNet-R101 (38.9)|39.9||||
    |FCOS-R50 (38.5)|FCOS-R101 (40.8)|42.2|||

* Stronger teachers:

    |Student|Teacher|MasKD|Config|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|
    |Faster RCNN-R50 (38.4)|Cascade Mask RCNN-X101 (45.6)|42.4||||
    |RetinaNet-R50 (37.4)|RetinaNet-X101 (41.0)|40.6||||
    |RepPoints-R50 (38.6)|RepPoints-R101 (44.2)|41.4|||
### Learning masks  
You can train your own mask tokens with the code provided in `mmdetection` folder. Please check [mmdetection/README.md](mmdetection/README.md) for detailed instructions.

### Semantic segmentation  
For semantic segmentation, please see `segmentation` folder.

## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation  