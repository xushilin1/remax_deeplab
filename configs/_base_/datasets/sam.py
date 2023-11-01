
dataset_type = 'SAMDataset'
data_root = 'data/sam/'

backend_args = None
image_size = (1024, 1024)

# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadJSONFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_point_coords=False),
    # dict(type=Resize, scale=image_size, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', by_mask=True, by_box=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'data_type')
    ),
    dict(type='GeneratePoint')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadJSONFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=False, with_mask=True, with_point_coords=False),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]

# dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    # batch_sampler=dict(type='AspectRatioBatchSampler')
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img=''),
        filter_cfg=None,
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = []
test_evaluator = val_evaluator
