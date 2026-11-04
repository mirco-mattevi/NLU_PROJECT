# Part 2.A - GPT2 for joint intent classification and slot filling (ATIS)

## Approach and changes with respect to LAB05

**Tuning procedure**

- One generic function (`run_tuning`) tunes a single hyperparameter: every candidate value is
  trained while the others stay fixed to the current best configuration, and a value is accepted
  only if it improves the mean dev slot F1 of the current best. It is called once per
  hyperparameter, in order: lr (baseline), d_model, n_heads, num_layers, ff_dim, and finally lr
  again (one value above and one below the current one), since the lr found for the tiny baseline
  may not suit the larger model.
- A lr that suits a small model can make a larger one unstable, which would make it look worse than
  it is. So after d_model (128, 256, 512) and num_layers (4, 6, 8) these larger models are also
  trained with the 2 lr below the current one. n_heads does not change the number of weights, and
  ff_dim showed no instability, so they are not retested.
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

**Slot evaluation**

- `pad` is excluded from the slot predictions in `eval_loop`. The loss already ignores pad positions,
  but the output layer still has a score for that label, so the model can predict it for a real
  token, especially in the first epochs. `conll.py` (unchanged from the lab) raises a `KeyError` on
  this label when it is the first token of a sentence or follows a slot label.
- The lab's `try/except` around `conll.evaluate` is removed. It set the F1 of the whole evaluation
  to 0 on any error, while `conll.py` already handles slot classes absent from the references, so
  any error left would be a real bug that should stop the run.

**Label ids**

- Slot and intent labels are sorted before assigning their ids (the lab uses a plain `set`). The
  order of a set of strings changes at every Python start, so the ids and the results of the same
  seed would change between sessions.

**Outputs**

- Every candidate appends a line to this README (dev/test slot F1, test intent accuracy, training
  time) and saves the loss curves of its best run in `plots/`.
- The best model of each tuning step is saved in `bin/` together with its configuration and the
  vocabularies needed to reload it. The configuration is needed because a wrong `n_heads` would
  load without errors (it does not change the shape of the weights) but give different results.

**Changes made after the first results**

- The `pad` exclusion, the removal of the `try/except` and the sorted labels were added after the
  lr, d_model, n_heads, num_layers and ff_dim results below, which were obtained with the lab's
  behavior. In the d_model to ff_dim runs the error occurred in two dev evaluations, both replaced
  by later epochs, and in no test evaluation. For the first lr tuning no log is available, but the
  small std of its test scores rules out a test score of 0.

## Results

[lr tuning] starting from {'lr': 0.001, 'd_model': 20, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: -1.0000)
lr=0.05 | dev slot F1: 0.9310 +- 0.0067 | test slot F1: 0.8866 +- 0.0027 | test intent acc: 0.9326 +- 0.0118 | 1.2 min
lr=0.01 | dev slot F1: 0.9212 +- 0.0166 | test slot F1: 0.8695 +- 0.0157 | test intent acc: 0.9147 +- 0.0269 | 1.9 min
lr=0.005 | dev slot F1: 0.9228 +- 0.0126 | test slot F1: 0.8685 +- 0.0098 | test intent acc: 0.9129 +- 0.0236 | 3.5 min
lr=0.001 | dev slot F1: 0.8679 +- 0.0213 | test slot F1: 0.8152 +- 0.0234 | test intent acc: 0.8905 +- 0.0124 | 7.6 min
lr=0.0005 | dev slot F1: 0.8544 +- 0.0203 | test slot F1: 0.7994 +- 0.0244 | test intent acc: 0.8802 +- 0.0101 | 13.2 min
lr=0.0001 | dev slot F1: 0.5768 +- 0.0115 | test slot F1: 0.5150 +- 0.0178 | test intent acc: 0.7881 +- 0.0289 | 16.0 min
--> best lr=0.05

[d_model tuning] starting from {'lr': 0.05, 'd_model': 20, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9310)
d_model=32 | dev slot F1: 0.9375 +- 0.0029 | test slot F1: 0.8932 +- 0.0055 | test intent acc: 0.9375 +- 0.0099 | 1.3 min d_model=64 | dev slot F1: 0.8194 +- 0.0394 | test slot F1: 0.7867 +- 0.0491 | test intent acc: 0.8291 +- 0.0403 | 1.2 min d_model=128 | dev slot F1: 0.3290 +- 0.1021 | test slot F1: 0.2884 +- 0.1065 | test intent acc: 0.7330 +- 0.0249 | 1.2 min d_model=256 | dev slot F1: 0.0690 +- 0.0185 | test slot F1: 0.0657 +- 0.0190 | test intent acc: 0.7406 +- 0.0100 | 1.1 min d_model=512 | dev slot F1: 0.0653 +- 0.0754 | test slot F1: 0.0595 +- 0.0650 | test intent acc: 0.7879 +- 0.0291 | 2.2 min
--> best d_model=32

[lr retuning for d_model=128] starting from {'lr': 0.05, 'd_model': 128, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9375)
lr=0.01 | dev slot F1: 0.9292 +- 0.0085 | test slot F1: 0.8893 +- 0.0091 | test intent acc: 0.9039 +- 0.0172 | 1.5 min
lr=0.005 | dev slot F1: 0.9429 +- 0.0048 | test slot F1: 0.8934 +- 0.0067 | test intent acc: 0.9234 +- 0.0119 | 1.9 min
lr=0.001 | dev slot F1: 0.9365 +- 0.0033 | test slot F1: 0.8808 +- 0.0061 | test intent acc: 0.9093 +- 0.0123 | 3.4 min
--> best lr=0.005

[lr retuning for d_model=256] starting from {'lr': 0.005, 'd_model': 256, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
lr=0.01 | dev slot F1: 0.8753 +- 0.0153 | test slot F1: 0.8313 +- 0.0161 | test intent acc: 0.8170 +- 0.0472 | 1.6 min
lr=0.005 | dev slot F1: 0.9115 +- 0.0137 | test slot F1: 0.8615 +- 0.0214 | test intent acc: 0.8385 +- 0.0779 | 1.4 min
lr=0.001 | dev slot F1: 0.9323 +- 0.0055 | test slot F1: 0.8812 +- 0.0029 | test intent acc: 0.9185 +- 0.0065 | 2.2 min
--> no lr improves the dev slot F1, kept lr=0.005

[lr retuning for d_model=512] starting from {'lr': 0.005, 'd_model': 512, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
lr=0.01 | dev slot F1: 0.5905 +- 0.1267 | test slot F1: 0.5450 +- 0.1300 | test intent acc: 0.7364 +- 0.0448 | 1.7 min
lr=0.005 | dev slot F1: 0.7991 +- 0.0356 | test slot F1: 0.7470 +- 0.0443 | test intent acc: 0.8092 +- 0.0169 | 1.8 min
lr=0.001 | dev slot F1: 0.9213 +- 0.0031 | test slot F1: 0.8760 +- 0.0101 | test intent acc: 0.8885 +- 0.0103 | 2.1 min
--> no lr improves the dev slot F1, kept lr=0.005

[n_heads tuning] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
n_heads=2 | dev slot F1: 0.9381 +- 0.0058 | test slot F1: 0.8857 +- 0.0069 | test intent acc: 0.9187 +- 0.0104 | 1.7 min
n_heads=4 | dev slot F1: 0.9370 +- 0.0039 | test slot F1: 0.8844 +- 0.0062 | test intent acc: 0.9187 +- 0.0159 | 1.4 min
n_heads=8 | dev slot F1: 0.9391 +- 0.0039 | test slot F1: 0.8821 +- 0.0070 | test intent acc: 0.9308 +- 0.0114 | 1.5 min
n_heads=16 | dev slot F1: 0.9386 +- 0.0036 | test slot F1: 0.8768 +- 0.0057 | test intent acc: 0.9306 +- 0.0030 | 1.6 min
--> no n_heads improves the dev slot F1, kept n_heads=1

[num_layers tuning] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
num_layers=2 | dev slot F1: 0.8662 +- 0.0581 | test slot F1: 0.8226 +- 0.0611 | test intent acc: 0.8770 +- 0.0335 | 1.2 min num_layers=3 | dev slot F1: 0.8584 +- 0.0628 | test slot F1: 0.8156 +- 0.0627 | test intent acc: 0.8757 +- 0.0505 | 1.5 min num_layers=4 | dev slot F1: 0.8711 +- 0.0411 | test slot F1: 0.8310 +- 0.0416 | test intent acc: 0.8934 +- 0.0346 | 1.7 min num_layers=6 | dev slot F1: 0.8721 +- 0.0314 | test slot F1: 0.8251 +- 0.0306 | test intent acc: 0.8961 +- 0.0241 | 1.7 min num_layers=8 | dev slot F1: 0.7745 +- 0.0601 | test slot F1: 0.7232 +- 0.0702 | test intent acc: 0.8325 +- 0.0465 | 1.6 min
--> no num_layers improves the dev slot F1, kept num_layers=1

[lr retuning for num_layers=4] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 4, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
lr=0.001 | dev slot F1: 0.9278 +- 0.0060 | test slot F1: 0.8755 +- 0.0032 | test intent acc: 0.9198 +- 0.0070 | 3.4 min
lr=0.0005 | dev slot F1: 0.9226 +- 0.0066 | test slot F1: 0.8630 +- 0.0062 | test intent acc: 0.9122 +- 0.0057 | 5.1 min
lr=0.0001 | dev slot F1: 0.8288 +- 0.0315 | test slot F1: 0.7879 +- 0.0290 | test intent acc: 0.8674 +- 0.0143 | 10.0 min
--> no lr improves the dev slot F1, kept lr=0.005

[lr retuning for num_layers=6] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 6, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
lr=0.001 | dev slot F1: 0.9301 +- 0.0094 | test slot F1: 0.8701 +- 0.0092 | test intent acc: 0.9288 +- 0.0139 | 3.7 min
lr=0.0005 | dev slot F1: 0.9156 +- 0.0108 | test slot F1: 0.8607 +- 0.0141 | test intent acc: 0.9247 +- 0.0125 | 5.5 min
lr=0.0001 | dev slot F1: 0.8605 +- 0.0207 | test slot F1: 0.8075 +- 0.0193 | test intent acc: 0.8703 +- 0.0113 | 14.3 min
--> no lr improves the dev slot F1, kept lr=0.005

[lr retuning for num_layers=8] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 8, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
lr=0.001 | dev slot F1: 0.9232 +- 0.0071 | test slot F1: 0.8713 +- 0.0064 | test intent acc: 0.9344 +- 0.0094 | 3.7 min
lr=0.0005 | dev slot F1: 0.9132 +- 0.0119 | test slot F1: 0.8556 +- 0.0058 | test intent acc: 0.9151 +- 0.0158 | 5.4 min
lr=0.0001 | dev slot F1: 0.8401 +- 0.0373 | test slot F1: 0.7927 +- 0.0307 | test intent acc: 0.8495 +- 0.0288 | 15.1 min
--> no lr improves the dev slot F1, kept lr=0.005

[ff_dim tuning] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
ff_dim=128 | dev slot F1: 0.9346 +- 0.0035 | test slot F1: 0.8866 +- 0.0024 | test intent acc: 0.9133 +- 0.0142 | 1.4 min ff_dim=256 | dev slot F1: 0.9381 +- 0.0048 | test slot F1: 0.8827 +- 0.0080 | test intent acc: 0.9135 +- 0.0108 | 1.6 min ff_dim=512 | dev slot F1: 0.9304 +- 0.0083 | test slot F1: 0.8815 +- 0.0101 | test intent acc: 0.9259 +- 0.0053 | 1.5 min ff_dim=1024 | dev slot F1: 0.9256 +- 0.0063 | test slot F1: 0.8712 +- 0.0130 | test intent acc: 0.9156 +- 0.0095 | 1.5 min
--> no ff_dim improves the dev slot F1, kept ff_dim=20

[final lr retuning] starting from {'lr': 0.005, 'd_model': 128, 'n_heads': 1, 'num_layers': 1, 'ff_dim': 20, 'dropout': 0.0} (dev slot F1 to beat: 0.9429)
lr=0.01 | dev slot F1: 0.9292 +- 0.0085 | test slot F1: 0.8893 +- 0.0091 | test intent acc: 0.9039 +- 0.0172 | 1.5 min
lr=0.001 | dev slot F1: 0.9365 +- 0.0033 | test slot F1: 0.8808 +- 0.0061 | test intent acc: 0.9093 +- 0.0123 | 3.4 min
--> no lr improves the dev slot F1, kept lr=0.005
