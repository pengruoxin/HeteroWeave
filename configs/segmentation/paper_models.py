"""Paper configurations for Oxford-IIIT Pets semantic segmentation."""

dataset = 'Oxford-IIIT Pets'
anchor = 'DeepLabV3-ResNet50'
seeds = [0, 1, 2]
training = dict(
    steps=10000,
    batch_size=8,
    resolution=256,
    optimizer='AdamW',
    learning_rate=1e-3,
    weight_decay=1e-4,
    checkpoint='last',
)

candidates = [
    dict(id='resnet50', position=None, components=[], total_blocks=1,
         gate=0.0, adapter_bottleneck=0, fusion='none', role='baseline'),
    dict(id='heteroweave_s1', position='layer3',
         components=['random_a', 'random_b'], total_blocks=3, gate=0.025,
         adapter_bottleneck=16, fusion='normalized_weighted_sum',
         role='capacity_control'),
    dict(id='heteroweave_s2', position='layer3',
         components=['dino', 'dino'], total_blocks=3, gate=0.025,
         adapter_bottleneck=16, fusion='normalized_weighted_sum',
         role='source_control'),
    dict(id='heteroweave_s3', position='layer3',
         components=['dino', 'swav'], total_blocks=3, gate=0.025,
         adapter_bottleneck=16, fusion='normalized_weighted_sum',
         role='main'),
    dict(id='heteroweave_s4', position='layer3', components=['dino'],
         total_blocks=2, gate=0.025, adapter_bottleneck=16,
         fusion='normalized_weighted_sum', role='main'),
]

