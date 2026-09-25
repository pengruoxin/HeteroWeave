"""Paper configurations for PASCAL VOC 2007 object detection."""

dataset = 'PASCAL VOC 2007 trainval'
detector = 'Faster R-CNN ResNet-50 FPN v2'
seeds = [0, 1, 2]
training = dict(
    steps=5000,
    batch_size=2,
    optimizer='SGD',
    learning_rate=5e-3,
    momentum=0.9,
    weight_decay=1e-4,
    min_size=384,
    max_size=640,
    checkpoint='last',
)

candidates = [
    dict(id='resnet50', position='none', components=[], total_blocks=1,
         gamma=0.0, role='baseline'),
    dict(id='heteroweave_d1', position='layer3', components=['byol'],
         total_blocks=2, gamma=0.001, role='main'),
    dict(id='heteroweave_d2', position='layer3', components=['byol'],
         total_blocks=2, gamma=0.0025, role='main'),
    dict(id='heteroweave_d3', position='layer4', components=['byol'],
         total_blocks=2, gamma=0.0025, role='main'),
]

