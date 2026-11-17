# predict-next-digit

A small PyTorch GRU that predicts the next digit (0-9) from a synthetic digit sequence. Data includes arithmetic, repeating-cycle, alternating, constant, Fibonacci-like, countdown, and random patterns.

## Run

```bash
python -m pip install -r requirements.txt
python train.py
python predict.py 1 3 5 7 9
```

Training generates fixed-seed 50,000/5,000/5,000 train/validation/test examples. It saves the best validation-accuracy checkpoint to `model.pt`, training curves to `curves.png`, and per-pattern results to `per_pattern.png`. Prediction accepts any sequence of at least three digits from 0 to 9.

## Test results

Results from the fixed-seed CPU run (best checkpoint selected at epoch 12; 115.4 seconds):

| Test set | GRU model | Repeat last | Most frequent |
|---|---:|---:|---:|
| Overall | 93.40% | 18.16% | 20.42% |
| Arithmetic | 100.00% | 0.00% | 0.00% |
| Repeating cycle | 84.20% | 8.60% | 20.23% |
| Alternating | 100.00% | 0.00% | 0.00% |
| Constant | 100.00% | 100.00% | 100.00% |
| Fibonacci-like | 100.00% | 8.89% | 11.36% |
| Countdown | 100.00% | 0.00% | 0.00% |
| Random | 10.48% | 13.54% | 13.97% |

## What it learns

The GRU can use several earlier digits to infer a pattern and project its continuation.
Repeating cycles and Fibonacci-like sequences require context beyond simply copying the last digit.
Constant and countdown examples are especially simple; the baselines provide useful comparisons.
The pattern families overlap, so ten observations do not always identify a unique generating rule.
Random sequences have unpredictable labels, limiting achievable accuracy to approximately chance.
