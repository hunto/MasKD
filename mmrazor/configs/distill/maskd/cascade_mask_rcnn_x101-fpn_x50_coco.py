_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_2x.py',
    '../../_base_/mmdet_runtime.py'
]

# model settings
student = dict(
    type='mmdet.FasterRCNN',
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
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))


teacher = dict(
    type='mmdet.CascadeRCNN',
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth'),
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
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=1,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                # NOTE: use `KDShared2FCBBoxHead` to discard loss computation
                type='KDShared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            # dict(
            #     type='Shared2FCBBoxHead',
            #     in_channels=256,
            #     fc_out_channels=1024,
            #     roi_feat_size=7,
            #     num_classes=80,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=[0., 0., 0., 0.],
            #         target_stds=[0.05, 0.05, 0.1, 0.1]),
            #     reg_class_agnostic=True,
            #     loss_cls=dict(
            #         type='CrossEntropyLoss',
            #         use_sigmoid=False,
            #         loss_weight=1.0),
            #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
            #                    loss_weight=1.0)),
            # dict(
            #     type='Shared2FCBBoxHead',
            #     in_channels=256,
            #     fc_out_channels=1024,
            #     roi_feat_size=7,
            #     num_classes=80,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=[0., 0., 0., 0.],
            #         target_stds=[0.033, 0.033, 0.067, 0.067]),
            #     reg_class_agnostic=True,
            #     loss_cls=dict(
            #         type='CrossEntropyLoss',
            #         use_sigmoid=False,
            #         loss_weight=1.0),
            #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        # mask_roi_extractor=dict(
        #     type='SingleRoIExtractor',
        #     roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
        #     out_channels=256,
        #     featmap_strides=[4, 8, 16, 32]),
        # mask_head=dict(
        #     type='FCNMaskHead',
        #     num_convs=4,
        #     in_channels=256,
        #     conv_out_channels=256,
        #     num_classes=80,
        #     loss_mask=dict(
        #         type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            # dict(
            #     assigner=dict(
            #         type='MaxIoUAssigner',
            #         pos_iou_thr=0.6,
            #         neg_iou_thr=0.6,
            #         min_pos_iou=0.6,
            #         match_low_quality=False,
            #         ignore_iof_thr=-1),
            #     sampler=dict(
            #         type='RandomSampler',
            #         num=512,
            #         pos_fraction=0.25,
            #         neg_pos_ub=-1,
            #         add_gt_as_proposals=True),
            #     pos_weight=-1,
            #     debug=False),
            # dict(
            #     assigner=dict(
            #         type='MaxIoUAssigner',
            #         pos_iou_thr=0.7,
            #         neg_iou_thr=0.7,
            #         min_pos_iou=0.7,
            #         match_low_quality=False,
            #         ignore_iof_thr=-1),
            #     sampler=dict(
            #         type='RandomSampler',
            #         num=512,
            #         pos_fraction=0.25,
            #         neg_pos_ub=-1,
            #         add_gt_as_proposals=True),
            #     pos_weight=-1,
            #     debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            )))


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
                        custom_mask_warmup=5000,
                        pretrained='https://github.com/hunto/MasKD/releases/download/v0.0.1/maskd_cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20220530.pth',
                        loss_weight=1,
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
                student_module='roi_head.bbox_head.fc_reg',
                teacher_module='roi_head.bbox_head.0.fc_reg',
                modules_with_student_inputs=[
                    dict(
                        student_module='roi_head.bbox_roi_extractor',
                        teacher_module='roi_head.bbox_roi_extractor.0',
                        same_indices=[1],)
                ],
                losses=[
                    dict(
                        type='mmdet.SmoothL1Loss',
                        name='loss_kd_roi_reg_head',
                        loss_weight=1,
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
optimizer = dict(paramwise_cfg=opt_paramwise_cfg)

