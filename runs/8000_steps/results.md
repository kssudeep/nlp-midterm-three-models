| Model | Params | Train Loss | Validation Loss | Training Time (s) | Generated Text Quality | Notes |
|---|---|---|---|---|---|---|
| Bigram | 4,225 | 2.4499 | 2.4829 | 8.4 | Low | No change from the 3,000-step run. |
| LSTM | 946,625 | 0.9956 | 1.5957 | ~249 (see note) | Medium | Overfits: val rose from 1.4903 (step 1,500) to 1.5957; train/val gap 0.60. |
| Transformer | 826,433 | 1.2150 | 1.5106 | 385.9 | Medium | Passes the LSTM's val loss at step ~4,750; train/val gap 0.30. |

Note on LSTM training time: the Mac went to sleep during the LSTM run, so the measured time
(22,481 s) includes several hours of sleep and is not a real training time. After the Mac woke,
steps 3,250 to 8,000 took 147.8 s (31.1 ms per step), which gives about 249 s for 8,000 steps.
That matches the clean 3,000-step run (90.0 s, 30.0 ms per step). Sleep does not change the
losses. The bigram and transformer times are measured directly.

Best validation loss during the run: Bigram 2.4821 (step 5000), LSTM 1.4903 (step 1500),
Transformer 1.5037 (step 7750).
