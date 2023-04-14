python -m torch.distributed.launch --nproc_per_node=4 \
    test.py \
    --model deeplab_mobile \
    --backbone mobilenetv2 \
    --data data/cityscapes/ \
    --save-dir work_dirs/test_images \
    --gpu-id 0,1,2,3 \
    --save-pred \
    --pretrained work_dirs/maskd_dv3-r101_dv3_mbv2/kd_deeplab_mobile_mobilenetv2_citys_best_model.pth
