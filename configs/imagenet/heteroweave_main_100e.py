_base_ = ['./_base_full_finetune_100e.py']

# HeteroWeave main representative: 18.258M parameters, 3.945G FLOPs.
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='DeRy',
        block_fixed=False,
        train_adapters_only=False,
        train_stem=True,
        base_channels=64,
        block_list=[
            [
                'swsl_resnext50_32x4d',
                'layer1.0',
                'layer1.2',
                'mytimm',
            ],
            ['resnet50', 'layer1.2', 'layer3.1', 'pytorch'],
            dict(
                type='CompositeBlock',
                branches=[
                    dict(
                        block=[
                            'resnet50', 'layer3.0', 'layer3.0', 'pytorch'
                        ],
                        input_adapter=None,
                        output_adapter=None),
                    dict(
                        block=[
                            'vit_small_patch16_224',
                            'blocks.6',
                            'blocks.6',
                            'mytimm',
                        ],
                        input_adapter=dict(
                            input_channel=512,
                            output_channel=384,
                            num_fc=0,
                            num_conv=1,
                            stride=1,
                            mode='cnn2vit'),
                        output_adapter=dict(
                            input_channel=384,
                            output_channel=1024,
                            num_fc=0,
                            num_conv=1,
                            stride=1,
                            mode='vit2cnn')),
                ],
                operator='sum',
                out_type='cnn'),
            [
                'vit_small_patch16_224',
                'blocks.7',
                'blocks.11',
                'mytimm',
            ],
        ],
        adapter_list=[
            dict(
                input_channel=256,
                output_channel=256,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=1024,
                output_channel=512,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2cnn'),
            dict(
                input_channel=1024,
                output_channel=384,
                stride=1,
                num_fc=0,
                num_conv=1,
                mode='cnn2vit'),
        ],
        out_indices=(3, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=384,
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
