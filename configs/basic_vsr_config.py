_base_ = 'default_runtime.py'

experiment_name = 'basicvsr'
work_dir = f'/content/drive/MyDrive/basicvsr/work_dirs/{experiment_name}'
save_dir = '/content/drive/MyDrive/basicvsr/work_dirs'

scale = 4
# /content/drive/MyDrive
# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRNet',
        mid_channels=64,
        num_blocks=30,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    reconstruct_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=scale)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='PackInputs')
]

data_root = '/content/drive/MyDrive/1THESIS/train'

train_dataloader = dict(
    num_workers=12,
    batch_size=4,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root=data_root,
        data_prefix=dict(img='train_sharp_bicubic/X4', gt='train_sharp'),
        depth=1,
        num_input_frames=5,
        pipeline=train_pipeline))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=300_000)

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.125)}))

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        save_optimizer=True,
        out_dir=save_dir,
        max_keep_ckpts=100,
        by_epoch=False),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

find_unused_parameters = True