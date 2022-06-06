norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='RepPointsDetector',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        num_outs=5),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'),
    train_cfg=dict(
        init=dict(
            assigner=dict(type='PointAssigner', scale=4, pos_num=1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        refine=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
dataset_type = 'RefCrowdDataset'
data_root = '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'refer_labels', 'att_labels',
            'att_label_weights'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'refer_labels'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type='RefCrowdDataset',
        ann_file=[
            '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/annotations/1011_json/refcrowdhuman2021_qhq_train_1011_h5id.json'
        ],
        img_prefix=
        '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/crowdhuman2021/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=[
                    'img', 'gt_bboxes', 'gt_labels', 'refer_labels',
                    'att_labels', 'att_label_weights'
                ])
        ]),
    val=dict(
        type='RefCrowdDataset',
        ann_file=[
            '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/annotations/1011_json/refcrowdhuman2021_qhq_val_1011_h5id.json'
        ],
        img_prefix=
        '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/crowdhuman2021/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'refer_labels'])
                ])
        ]),
    test=dict(
        type='RefCrowdDataset',
        ann_file=
        '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/annotations/1011_json/refcrowdhuman2021_qhq_val_1011_h5id.json',
        img_prefix=
        '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/crowdhuman2021/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'refer_labels'])
                ])
        ]))
evaluation = dict(interval=1, metric='Top1Acc')
optimizer = dict(
    type='SGD',
    lr=0.002,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            dynamic_fcs=dict(lr_mult=10.0, decay_mult=1.0),
            rnn_encoder=dict(lr_mult=10.0, decay_mult=1.0),
            contra=dict(lr_mult=10.0, decay_mult=1.0),
            lang=dict(lr_mult=10.0, decay_mult=1.0),
            vis=dict(lr_mult=10.0, decay_mult=1.0),
            word=dict(lr_mult=10.0, decay_mult=1.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
load_from = '/data1/QiuHeqian/refcrowdhuman_qhq/mmdetection-refcrowd-cvpr/pretrained_models/reppoints_moment_r101_fpn_gn-neck+head_2x_coco_20200329-4fbc7310.pth'
work_dir = './work_dirs/reppoints_r101_v1'
gpu_ids = range(0, 8)
