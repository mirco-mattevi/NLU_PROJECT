# Part 2.A - GPT2 for joint intent classification and slot filling (ATIS)

## Approach and changes with respect to LAB05

**Tuning procedure**

- One generic function (`run_tuning`) tunes a single hyperparameter: every candidate value is
  trained while the others stay fixed to the current best configuration, and a value is accepted
  only if it improves the mean dev slot F1 of the current best. It is called once per
  hyperparameter, in order: lr (baseline), d_model, n_heads, num_layers, ff_dim, and finally lr
  again, since the lr found for the tiny baseline may not suit the larger model.
- Every candidate is trained 5 times from scratch (`multiple_runs`) and results are reported as
  mean +- std, since single runs on ATIS are noisy.
- Candidates are compared on the **dev** slot F1, the test scores are only reported (in the lab
  `multiple_runs` returned only test scores).
- Run _i_ of every candidate uses seed _i_: all candidates see the same initializations and batch
  orders, so differences between them are due to the hyperparameter rather than to the seed, and
  results are reproducible.
- Dropout is explicitly set to 0.0 for experiments 0 and 1.

**Early stopping**

- In the lab, patience is decreased when the dev slot F1 does not improve (in the multiple-runs
  cell it is evaluated every 5 epochs and never reset). With GPT2 this stops training too early:
  in the first epochs the model predicts only "O" for every token, so slot F1 stays at ~0 for
  several epochs while the model is still learning (the dev loss keeps decreasing).
- Here patience (3) is reset when either the dev slot F1 or the dev loss improves, while the model
  is still selected on the dev slot F1. Baseline (d_model=20), single run, dev slot F1:

  | lr     | patience on F1 only         | patience on F1 or dev loss |
  | ------ | --------------------------- | -------------------------- |
  | 0.001  | 0.18 (stops after 4 epochs) | 0.893 (113 epochs)         |
  | 0.0001 | 0.00 (stops after 4 epochs) | 0.566 (200 epochs)         |

- The best-dev model is kept and evaluated on the test set (the lab evaluated the last model).
- The best F1 starts from -1, so a model is always kept even if the dev F1 never leaves 0.

**Outputs**

- Every candidate appends a line to this README (dev/test slot F1, test intent accuracy, training
  time) and saves the loss curves of its best run in `plots/`.
- The best model of each tuning step is saved in `bin/` together with the vocabularies needed to
  reload it.

## Results

[lr tuning] starting from {'lr': 0.001, 'd_model': 20, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: -1.0000)
lr=0.05 | dev slot F1: 0.9308 +- 0.0064 | test slot F1: 0.8870 +- 0.0089 | test intent acc: 0.9268 +- 0.0098 | 1.2 min
lr=0.01 | dev slot F1: 0.9286 +- 0.0065 | test slot F1: 0.8786 +- 0.0080 | test intent acc: 0.9362 +- 0.0096 | 2.0 min
lr=0.005 | dev slot F1: 0.9226 +- 0.0030 | test slot F1: 0.8721 +- 0.0093 | test intent acc: 0.9274 +- 0.0023 | 3.2 min
lr=0.001 | dev slot F1: 0.8845 +- 0.0226 | test slot F1: 0.8283 +- 0.0268 | test intent acc: 0.8907 +- 0.0134 | 8.1 min
lr=0.0005 | dev slot F1: 0.8540 +- 0.0112 | test slot F1: 0.8054 +- 0.0127 | test intent acc: 0.8768 +- 0.0180 | 11.0 min
--> best lr=0.05 | dev slot F1: 0.9308 +- 0.0064 | test slot F1: 0.8870 +- 0.0089 | test intent acc: 0.9268 +- 0.0098

"0.05 and 0.01 are within one std; 0.01 kept for the architecture search since 0.05 is likely too high for larger models; 0.05 is re-tested in the final lr tuning."
