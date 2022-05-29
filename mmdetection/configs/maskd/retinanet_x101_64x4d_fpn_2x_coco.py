_base_ = '../retinanet/retinanet_x101_64x4d_fpn_2x_coco.py'

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
        checkpoint='https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth'
    ),
)

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=2000)
lr_config = dict(_delete_=True, policy='CosineAnnealing', min_lr=1e-6)
optimizer = dict(_delete_=True, type='Adam', lr=0.01, weight_decay=0.001)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='bbox')

