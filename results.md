| Model | Params | Train Loss | Validation Loss | Training Time (s) | Generated Text Quality | Notes |
|---|---|---|---|---|---|---|
| Bigram | 4,225 | 2.4532 | 2.4835 | 3.1 | Low | Flat at ~2.48 from step 1,000 on. Common letter pairs, almost no real words. |
| LSTM | 946,625 | 1.1335 | 1.5091 | 90.0 | Medium | Lowest val loss at this budget. Best val 1.4898 at step 1,500, then overfits. |
| Transformer | 826,433 | 1.3316 | 1.5692 | 141.8 | Medium | Val loss still falling at step 3,000 (not converged). More invented words than the LSTM. |
