_base_ = ['../imagenet/dery_baseline_100e.py']

# Capacity-matched parallel composition: 29.18M parameters, 4.70G FLOPs.
model = dict(
    backbone=dict(
        block_list=[
            ['swsl_resnext50_32x4d', 'layer1.0', 'layer1.2', 'mytimm'],
            ['resnet50', 'layer1.2', 'layer3.1', 'pytorch'],
            dict(
                type='CompositeBlock',
                branches=[
                    dict(
                        block=['resnet50', 'layer2.2', 'layer2.2', 'pytorch'],
                        input_adapter=None,
                        output_adapter=None),
                    dict(
                        block=[
                            'regnet_y_1_6gf',
                            'trunk_output.block2.block2-3',
                            'trunk_output.block3.block3-10',
                            'pytorch',
                        ],
                        input_adapter=dict(
                            input_channel=512, output_channel=120,
                            num_fc=0, num_conv=1, stride=1, mode='cnn2cnn'),
                        output_adapter=dict(
                            input_channel=336, output_channel=512,
                            num_fc=0, num_conv=1, stride=1, mode='cnn2cnn')),
                ],
                operator='sum',
                out_type='cnn'),
            [
                'swin_small_patch4_window7_224', 'stages.2.blocks.16',
                'stages.3.blocks.1', 'mmcv',
            ],
        ]))

