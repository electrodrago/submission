_base_ = 'default_runtime.py'

experiment_name = 'test_real_cleanvsr'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs/'

scale = 4

model = dict(
    type='RealCleanVSR',
    generator=dict(
        type='RealCleanVSRNet',
        mid_channels=64, 
        num_blocks=12,
        num_clean_blocks=15,
        max_residue_magnitude=10,
        spynet_pretrained='./spynet/spynet.pth'),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    dynamic_clean_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    is_use_sharpened_gt_in_pixel=True,
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:08d}.png'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

demo_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='PackInputs')
]

data_root = "data"

test_dataloader = dict(
    num_workers=1,
    batch_size=1,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicFramesDataset',
        metainfo=dict(dataset_type='reds', task_name='vsr'),
        data_root= data_root,
        data_prefix=dict(img='path/LR', gt='path/HR'),
        pipeline=test_pipeline))


test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='NIQE', input_order='CHW', convert_to='Y'),
        dict(type='PSNR'),
        dict(type='SSIM'),
    ])

test_cfg = dict(type='MultiTestLoop')

# NO learning policy

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5000,
        save_optimizer=True,
        out_dir=save_dir,
        max_keep_ckpts=10,
        save_best='PSNR',
        rule='greater',
        by_epoch=False),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

model_wrapper_cfg = dict(
    type='MMSeparateDistributedDataParallel',
    broadcast_buffers=False,
    find_unused_parameters=False)

# VisBackend
vis_backends = [dict(type='LocalVisBackend')]
# Visualizer
visualizer = dict(
    type='ConcatImageVisualizer',
    vis_backends=vis_backends,
    fn_key='gt_path',
    img_keys=['pred_img'],
    bgr2rgb=True)
# VisualizationHook
custom_hooks = [dict(type='BasicVisualizationHook', interval=1)]

