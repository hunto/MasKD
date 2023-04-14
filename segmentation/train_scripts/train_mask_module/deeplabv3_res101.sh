python -m torch.distributed.launch --nproc_per_node=8 train_mask_module.py \
    --model deeplabv3 \
    --backbone resnet101 \
    --data data/cityscapes/ \
    --save-dir work_dirs/mask/dv3-r101 \
    --log-dir work_dirs/mask/dv3-r101 \
    --gpu-id 0,1,2,3,4,5,6,7 \
    --pretrained ckpts/deeplabv3_resnet101_citys_best_model.pth
