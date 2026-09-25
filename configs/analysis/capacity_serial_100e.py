_base_ = ['../imagenet/dery_baseline_100e.py']

# Capacity-matched serial extension: 29.18M parameters, 4.69G FLOPs.
model = dict(
    backbone=dict(
        capacity_control=dict(
            in_channels=768, hidden_channels=610, activation='relu')))

