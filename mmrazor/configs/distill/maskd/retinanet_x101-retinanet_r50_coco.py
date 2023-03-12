_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_2x.py',
    '../../_base_/mmdet_runtime.py'
]

# model settings
student = dict(
    type='mmdet.RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        init_cfg=dict(type='Pretrained', prefix='neck', checkpoint='token_ckpt/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth')),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        init_cfg=dict(type='Pretrained', prefix='bbox_head', checkpoint='token_ckpt/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth')
        ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

teacher = dict(
    type='mmdet.RetinaNet',
    init_cfg=dict(type='Pretrained', checkpoint='token_ckpt/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth'),
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='neck',
                teacher_module='neck',
                losses=[
                    dict(
                        type='MasKDLoss',
                        name='loss_maskd_fpn',
                        channels=[256, 256, 256, 256, 256],
                        num_tokens=6,
                        weight_mask=True,
                        custom_mask=True,
                        # custom_mask_warmup=7330,
                        custom_mask_warmup=8500,
                        pretrained='https://github.com/Gumpest/MasKD/releases/download/v0.0.3/maskd_retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.ckpt',
                        loss_weight=7,
                    ),
                ],
                align_module=dict(
                    type='conv2d',
                    num_modules=5,
                    student_channels=256,
                    teacher_channels=256,
                )
            ),
            dict(
                student_module='bbox_head.retina_reg',
                teacher_module='bbox_head.retina_reg',
                losses=[
                    dict(
                        type='mmdet.SmoothL1Loss',
                        name='loss_kd_reg_head',
                        loss_weight=1, # 1 / 5
                    )
                ]
            ),
        ]),
)

find_unused_parameters = True

opt_paramwise_cfg = dict(
    custom_keys={
        'distiller.teacher': dict(lr_mult=0, decay_mult=0),
        'distiller.losses': dict(lr_mult=0, decay_mult=0),
    }
)

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001, paramwise_cfg=opt_paramwise_cfg)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(warmup_iters=1000)
