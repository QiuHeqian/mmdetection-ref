model = dict(
    type='FCOS',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron/resnet101_caffe')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_att_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.0)),
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
dataset_type = 'RefCrowdDataset'
data_root = '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/'
img_norm_cfg = dict(
    mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(
        type='Normalize',
        mean=[102.9801, 115.9465, 122.7717],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
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
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'refer_labels'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RefCrowdDataset',
        ann_file=[
            '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/annotations/220331/refcrowd2021_revised_220331_h5id_whole_v2_select_86_attref_train.json'
        ],
        img_prefix=
        '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/crowdhuman2021/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[102.9801, 115.9465, 122.7717],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
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
            '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/annotations/220331/refcrowd2021_revised_220331_h5id_whole_v2_select_86_attref_val.json'
        ],
        img_prefix=
        '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/crowdhuman2021/',
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
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img', 'refer_labels'])
                ])
        ]),
    test=dict(
        type='RefCrowdDataset',
        ann_file=
        '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/annotations/220331/refcrowd2021_revised_220331_h5id_whole_v2_select_86_attref_test.json',
        img_prefix=
        '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/refcrowd_dataset/crowdhuman2021/',
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
                        mean=[102.9801, 115.9465, 122.7717],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
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
        bias_lr_mult=2.0,
        bias_decay_mult=0.0,
        custom_keys=dict({
            'dynamic_fcs':
            dict(lr_mult=10.0, decay_mult=1.0),
            'rnn_encoder':
            dict(lr_mult=10.0, decay_mult=1.0),
            'crowd':
            dict(lr_mult=10.0, decay_mult=1.0),
            'att':
            dict(lr_mult=10.0, decay_mult=1.0),
            'bbox_head.conv_cls':
            dict(lr_mult=10.0, decay_mult=1.0)
        })))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
resume_from = None
workflow = [('train', 1)]
load_from = '/data1/data/QiuHeqian/code/refcrowdhuman_qhq/mmdetection-master-ACMMM2022/pretrained_models/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco-511424d6.pth'
work_dir = './work_dirs/fcos_r101_v60'
gpu_ids = range(0, 1)
