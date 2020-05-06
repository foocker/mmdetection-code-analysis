model = dict(
    type='HeatMap',
    backbone=dict(
        type='ResNett',
        depth=18,
        in_channels=3, 
        dcn=None
    ),
    head=dict(
        type='HeatHead',
        heads=dict(
            hm=1,   # see opts.py
            wh=2,
            reg=2),
        head_conv=64, 
        dcn=None
    )
)