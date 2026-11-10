"""Paper configurations for the Flickr8K retrieval experiment."""

clip_model = 'openai/clip-vit-base-patch16'
dataset = 'jxie/flickr8k'
seeds = [0, 1, 2]
training = dict(
    steps=1000,
    batch_size=32,
    learning_rate=3e-3,
    weight_decay=1e-4,
    alignment_weight=0.1,
)

candidates = [
    dict(id='clip', block=None, donors=[], gate=0.0, adapter='none',
         width=0, fusion='none', tag='baseline'),
    dict(id='heteroweave_r1', block=10, donors=['mae'], gate=0.05,
         adapter='bottleneck', width=4, fusion='sum', tag='main'),
    dict(id='heteroweave_r2', block=10, donors=['mae', 'dino'], gate=0.025,
         adapter='bottleneck', width=16, fusion='sum', tag='main'),
    dict(id='heteroweave_r3', block=10, donors=['mae', 'mae'], gate=0.025,
         adapter='bottleneck', width=16, fusion='sum', tag='main'),
    dict(id='wider_adapter', block=10, donors=['adapter_only'], gate=0.025,
         adapter='bottleneck', width=32, fusion='sum', tag='control'),
    dict(id='parallel_adapters', block=10,
         donors=['adapter_only', 'adapter_only'], gate=0.025,
         adapter='bottleneck', width=16, fusion='sum', tag='control'),
]

