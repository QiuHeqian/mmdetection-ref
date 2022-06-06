# dataset settings
dataset_type = 'RefCrowdDataset'
data_root = '/data1/QiuHeqian/refcrowdhuman_qhq/refcrowd_dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0), #qhq,2020/08/10
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','refer_labels','att_labels','att_label_weights']), #,'word_weights']),  #qhq, add 'refer_labels' #qhq, add 'word_weights', 2020/09/24
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
        ann_file=[data_root+'annotations/1011_json/refcrowdhuman2021_qhq_train_1011_h5id.json'],
        img_prefix=data_root + 'crowdhuman2021/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=[data_root+'annotations/1011_json/refcrowdhuman2021_qhq_val_1011_h5id.json'],
        img_prefix=data_root + 'crowdhuman2021/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'annotations/1011_json/refcrowdhuman2021_qhq_val_1011_h5id.json',
        img_prefix=data_root + 'crowdhuman2021/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='Top1Acc')
