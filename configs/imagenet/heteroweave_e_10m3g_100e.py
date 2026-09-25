_base_ = ['./_base_full_finetune_100e.py']

# HeteroWeave-E (10M/3G): 7.711M parameters, 1.615G FLOPs.
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        train_adapters_only=False,
        train_stem=True,
        base_channels=64,
        block_list=[
            ['resnet50', 'layer1.0', 'layer1.2', 'pytorch'],
            [
                'regnet_y_1_6gf',
                'trunk_output.block2.block2-0',
                'trunk_output.block2.block2-2',
                'pytorch',
            ],
            [
                'regnet_y_800mf',
                'trunk_output.block3.block3-3',
                'trunk_output.block3.block3-7',
                'pytorch',
            ],
            [
                'regnet_y_1_6gf',
                'trunk_output.block3.block3-15',
                'trunk_output.block4.block4-1',
                'pytorch',
            ],
        ],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=48,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=120,
                output_channel=320,
                stride=2,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=320,
                output_channel=336,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
        ],
        out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=888,
        loss=dict(
            type='LabelSmoothLoss',
            label_smooth_val=0.1,
            num_classes=1000,
            reduction='mean',
            loss_weight=1.0),
        topk=(1, 5),
        cal_acc=False),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.1, num_classes=1000, prob=0.5),
        dict(type='BatchCutMix', alpha=1.0, num_classes=1000, prob=0.5),
    ]))
