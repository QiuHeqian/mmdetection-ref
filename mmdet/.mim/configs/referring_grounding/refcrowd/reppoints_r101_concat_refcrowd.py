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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        norm_cfg=norm_cfg,
        num_outs=5),
    bbox_head=dict(
        type='RepPointsHead',
        num_classes=1,#refer
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        norm_cfg=norm_cfg,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        lang_encoder=dict(
            type='RNNEncoder',
            vocab_size=2022,
            word_embedding_size=512,
            word_vec_size=512,
            hidden_size=512,
            bidirectional=True,
            word_drop_out=0.2,
            rnn_drop_out=0.0,
            rnn_num_layers=1,
            rnn_type='lstm',
            variable_lengths=True
        ),
        multimodal_fusion=dict(
            type='VL_Concat',
            hidden_size=512,
            num_input_channels=256,
            num_output_channels=256,
            num_featmaps=5
        ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='SmoothL1Loss', beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
        transform_method='moment'),
    # training and testing settings
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

#dataset settings
dataset_type = 'RefCrowdDataset'
data_root = 'data/RefCrowd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','refer_labels','att_labels','att_label_weights']),
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
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img','refer_labels']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=[data_root+'annotations/train.json'],
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=[data_root + 'annotations/val.json'],
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='Top1Acc')

optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001,
                  paramwise_cfg=dict(
                  custom_keys={
                      'lang_encoder': dict(lr_mult=10.0, decay_mult=1.0),
                      'multimodal_fusion': dict(lr_mult=10.0, decay_mult=1.0),
                      'bbox_head.conv_cls':dict(lr_mult=10.0, decay_mult=1.0)
                  })
                 )

optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = None
resume_from = None
workflow = [('train', 1)]
load_from='pretrained_models/coco_train_minus_refer/reppoints_r101.pth'
