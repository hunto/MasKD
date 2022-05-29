_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py'

find_unused_parameters = True

custom_hooks = [
    dict(
        type='MasKDHook',
        priority='HIGHEST',
        module_name='neck',
        channels=[256, 256, 256, 256, 256],
        num_tokens=6,
        weight_mask=True,
    )
]

model = dict(
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_2x_coco/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
    ),
)

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=2000)
lr_config = dict(_delete_=True, policy='CosineAnnealing', min_lr=1e-6)
optimizer = dict(lr=0.1)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='bbox')
