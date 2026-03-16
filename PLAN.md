# Step 4 — Training & Mitigating Overfitting — Implementation Plan

## Audit: What Already Exists (Step 3)

| Technique | Current State |
|---|---|
| LSTM dropout (0.2) | ✅ between stacked layers |
| Dense dropout (0.3) | ✅ in head |
| L2 regularisation (weight_decay=1e-5) | ✅ via AdamW |
| Early stopping (patience=10) | ✅ with best-checkpoint restore |
| Gradient clipping (max_norm=1.0) | ✅ per-step |
| LR scheduling (ReduceLROnPlateau) | ✅ factor=0.5 |
| Chronological train/val split | ✅ last 20% held out |

## What Step 4 Adds

### 4(a) Enhanced Regularisation — `src/model.py` changes

**Add input dropout** before the LSTM so that random features are zeroed each forward pass, forcing the network to learn robust representations that don't depend on any single feature.

**Add batch normalisation** (`nn.BatchNorm1d`) in the dense head to stabilise internal activations and improve convergence.

Changes to `SpreadLSTM`:
- New `input_dropout` parameter (default 0.1) + `nn.Dropout` applied to input `x` before LSTM
- Replace dense head `Linear → ReLU → Dropout → Linear` with `Linear → BatchNorm1d → ReLU → Dropout → Linear`
- New config constants: `INPUT_DROPOUT = 0.1`

### 4(b) Enhanced Early Stopping — `src/trainer.py` changes

Add a **min_delta** threshold so tiny improvements (noise) don't reset the patience counter.

Add **overfitting gap tracking**: compute and report (train_loss − val_loss) each epoch so we can see exactly when the gap diverges.

Changes to `Trainer`:
- New `min_delta` parameter (default 1e-4): val loss must improve by at least this much
- `EpochMetrics` gains `overfit_gap` field (val_loss − train_loss)
- `TrainingResult` gains `overfit_gap_at_best` field
- Update `fit()` improvement check: `if val_loss < self._best_val_loss - self.min_delta:`

Config: `EARLY_STOPPING_MIN_DELTA = 1e-4`

### 4(c) Walk-Forward Cross-Validation — **new** `src/walk_forward_cv.py`

Replace the single holdout with a strict **expanding-window time-series CV**:

```
Fold 1: Train [2020-01 .. 2021-12] → Val [2022-01 .. 2022-06]
Fold 2: Train [2020-01 .. 2022-06] → Val [2022-07 .. 2022-12]
Fold 3: Train [2020-01 .. 2022-12] → Val [2023-01 .. 2023-06]
Fold 4: Train [2020-01 .. 2023-06] → Val [2023-07 .. 2023-12]
```

Each fold:
1. Trains a fresh model from scratch
2. Applies all regularisation from 4(a)
3. Uses early stopping from 4(b)
4. Records per-fold metrics (MAE, RMSE, R²)

After all folds, aggregate metrics (mean ± std) give a robust, leak-free estimate of generalisation performance.

New module exports:
- `WalkForwardFold` dataclass (fold_id, train_period, val_period, metrics)
- `WalkForwardResult` dataclass (folds list, aggregate metrics)
- `generate_walk_forward_folds()` — splits PreparedArrays into chronological folds
- `run_walk_forward_cv()` — orchestrates multi-fold training

Config additions:
- `CV_N_FOLDS = 4`
- `CV_VAL_MONTHS = 6`  (validation window size)

### Pipeline — **new** `run_step4.py`

Orchestrates the full Step 4 pipeline:
1. Load Step 2 prepared data
2. Print enhanced regularisation summary (input dropout, batch norm, etc.)
3. Run walk-forward cross-validation (4 folds)
4. Report per-fold and aggregate metrics
5. Final holdout training with enhanced model (same as Step 3 but with 4a/4b improvements)
6. Generate submission.csv
7. Save artefacts: `spread_lstm_step4.pt`, `cv_results.csv`, updated `training_history.csv`

### Visualisations — `src/visualisation.py` additions

3 new plots:
1. `plot_cv_fold_metrics()` — bar chart comparing MAE/RMSE/R² across CV folds
2. `plot_overfitting_analysis()` — train vs val loss with overfitting gap shaded
3. `plot_regularisation_comparison()` — side-by-side training curves: Step 3 (baseline) vs Step 4 (enhanced)

### Tests — **new** `tests/test_step4.py`

~20 tests covering:
- `TestEnhancedModel` (6): input dropout effect, batch norm presence, output shape unchanged, parameter count change, dropout rate configurability, backward compatibility
- `TestEnhancedTrainer` (4): min_delta logic, overfit_gap tracking, gap recorded in history, early stop with min_delta
- `TestWalkForwardCV` (8): correct fold count, expanding training windows, no future leakage per fold, val periods non-overlapping, each fold runs training, aggregate metrics calculated, fold chronological order, minimum fold size
- `TestVisualisations` (2): new plot functions produce files

### File Changes Summary

| File | Action | Description |
|---|---|---|
| `config.py` | MODIFY | Add INPUT_DROPOUT, EARLY_STOPPING_MIN_DELTA, CV_N_FOLDS, CV_VAL_MONTHS |
| `src/model.py` | MODIFY | Add input dropout + batch norm to SpreadLSTM |
| `src/trainer.py` | MODIFY | Add min_delta + overfit_gap tracking |
| `src/walk_forward_cv.py` | CREATE | Walk-forward expanding-window CV engine |
| `src/visualisation.py` | MODIFY | Add 3 new plot functions |
| `src/__init__.py` | MODIFY | Export new Step 4 modules |
| `run_step4.py` | CREATE | Step 4 pipeline orchestrator |
| `tests/test_step4.py` | CREATE | ~20 unit tests |
| `README.md` | MODIFY | Add Step 4 documentation |
