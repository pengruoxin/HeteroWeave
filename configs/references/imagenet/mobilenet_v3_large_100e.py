_base_ = ['./_base_reference_100e.py']

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TIMMBackbone', model_name='mobilenetv3_large_100',
        pretrained=False),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead', num_classes=1000, in_channels=1280,
        loss=dict(type='LabelSmoothLoss', label_smooth_val=0.1,
                  num_classes=1000, reduction='mean', loss_weight=1.0),
        topk=(1, 5), cal_acc=False))
