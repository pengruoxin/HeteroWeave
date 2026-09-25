_base_ = ['./_base_100e.py']

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ModelStitchingBaseline',
        variant='high_r18_to_r50',
        pretrained=True,
        freeze_sources=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead', num_classes=1000, in_channels=2048,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1,
                  num_classes=1000, reduction='mean', loss_weight=1.0),
        topk=(1, 5), cal_acc=False),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.1, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5),
    ]))
