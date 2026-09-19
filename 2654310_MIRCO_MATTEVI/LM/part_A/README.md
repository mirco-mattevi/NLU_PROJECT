**[0 - baseline, learning rate tuning]**

lr=0.1 | dev PPL: 97.17 | test PPL: 85.09
lr=0.05 | dev PPL: 78.00 | test PPL: 69.23
lr=0.01 | dev PPL: 57.31 | test PPL: 50.38
lr=0.005 | dev PPL: 54.82 | test PPL: 48.14
lr=0.001 | dev PPL: 51.28 | test PPL: 44.98
lr=0.0005 | dev PPL: 51.73 | test PPL: 45.25
lr=0.0001 | dev PPL: 56.20 | test PPL: 48.74

**[1 - hyperparameter optimization]**

d_model=64 | dev PPL: 46.36 | test PPL: 40.90
d_model=128 | dev PPL: 45.51 | test PPL: 40.73
d_model=256 | dev PPL: 46.07 | test PPL: 41.48

n_heads=2 | dev PPL: 44.13 | test PPL: 39.49
n_heads=4 | dev PPL: 44.50 | test PPL: 39.77
n_heads=8 | dev PPL: 44.60 | test PPL: 39.84

num_layers=2 | dev PPL: 43.67 | test PPL: 38.94
num_layers=4 | dev PPL: 42.26 | test PPL: 37.96
num_layers=6 | dev PPL: 41.74 | test PPL: 37.69

ff_dim=256 | dev PPL: 40.21 | test PPL: 36.23
ff_dim=512 | dev PPL: 39.58 | test PPL: 36.04

--> best after experiment 1: d_model=128 n_heads=2 num_layers=6 ff_dim=512 | dev PPL: 39.58 | test PPL: 36.04

**[1b - lr sensitivity spot-check (lr=0.001 vs lr=0.0005)]**

d_model=256 (n_heads=1, num_layers=1, ff_dim=20) | lr=0.0005 | dev PPL: 48.33 | test PPL: 42.92 (round winner at lr=0.001, d_model=128: 45.51) -> does NOT improve on the lr=0.001 reference

num_layers=6 (d_model=128, n_heads=2, ff_dim=20) | lr=0.0005 | dev PPL: 45.54 | test PPL: 40.64 (round winner at lr=0.001, num_layers=6 itself: 41.74) -> does NOT improve on the lr=0.001 reference
