python -m torch.distributed.launch --nproc_per_node=1 eval_vis_mask.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --data data/cityscapes/ \
    --save-dir work_dirs/mask/dv3-r101/vis \
    --gpu-id 0 \
    --pretrained ckpts/deeplabv3_resnet101_citys_best_model.pth \
    --pretrained-mask work_dirs/mask/dv3-r101/deeplabv3_resnet101_citys_best_model.pth
