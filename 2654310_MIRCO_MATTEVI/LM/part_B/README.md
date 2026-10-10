[0 - learning rate tuning] rank=8 alpha=16
lr=0.0001 | dev PPL: 22.60 | test PPL: 20.49
lr=0.0003 | dev PPL: 21.88 | test PPL: 19.88
lr=0.001 | dev PPL: 22.82 | test PPL: 20.64

[1 - rank tuning] lr=0.0003 alpha=2\*rank
rank=4 alpha=8 | dev PPL: 23.02 | test PPL: 20.92
rank=8 alpha=16 | dev PPL: 21.69 | test PPL: 19.77
rank=16 alpha=32 | dev PPL: 20.95 | test PPL: 19.07
rank=32 alpha=64 | dev PPL: 20.39 | test PPL: 18.53
rank=64 alpha=128 | dev PPL: 20.18 | test PPL: 18.48
