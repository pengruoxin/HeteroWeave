_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    '../_base_/default_runtime.py',
]

# Comparable ImageNet-1K recipe: 8 GPUs x 128 images, global batch 1024.
runner = dict(type='EpochBasedRunner', max_epochs=100)
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=4,
    train=dict(data_prefix='data/imagenet/train'),
    val=dict(
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt'),
    test=dict(
        data_prefix='data/imagenet/val',
        ann_file='data/imagenet/meta/val.txt'))

evaluation = dict(interval=10, metric='accuracy')
checkpoint_config = dict(interval=5, max_keep_ckpts=2)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={'.cls_token': dict(decay_mult=0.0)}))

fp16 = dict(loss_scale=512.0)
dist_params = dict(backend='nccl')
