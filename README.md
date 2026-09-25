# From Data Leakage to Architecture Optimization: Design Principles and Empirical Study of Causal Transformers for CNC Predictive Maintenance

Data processing, model architecture, ablation studies, and statistical validation code for a conference paper proposing two transferable design principles for causal-attention sequence models in predictive maintenance.

## 1. Paper & Conference

| Item | Detail |
|---|---|
| Title | From Data Leakage to Architecture Optimization: Design Principles and Empirical Study of Causal Transformers for CNC Predictive Maintenance |
| Authors | Li Yang Chou¹* (周理陽), Peng Kai Wang² |
| Affiliations | ¹Department of Mechanical Engineering, National Central University; ²Bachelor Degree Program of Applied Science and Technology, National Taiwan University of Science and Technology |
| Keywords | Predictive Maintenance, Causal Transformer, Multi-Task Learning, Ablation Study, CNC Machine Tools |

| Conference | Detail |
|---|---|
| Full name | 中華民國品質學會第62屆年會暨2026國際品質管理研討會 — Chinese Society for Quality (CSQ) 62nd Annual Meeting & 2026 International Symposium on Quality Management (ISQM 2026) |
| Organizers | Chinese Society for Quality (CSQ), Department of Industrial Engineering and Management, Yuan Ze University |
| Date | 7 November 2026 |
| Venue | You Yang Hall, Yuan Ze University, Taoyuan, Taiwan |
| Official website | https://sites.google.com/view/isqm2026 |
| Status | Accepted |

### Abstract

Motivated by two design pitfalls commonly overlooked when applying causal-attention sequence models to predictive maintenance, this study proposes and validates two corresponding design principles using CNC tool-wear prediction as a case study. First, a window-alignment principle: when an input feature is self-referential with the prediction target, the target should be aligned to the next time step (t+1) rather than the window's own last step, structurally preventing data leakage. Second, a pooling-strategy principle: causal-masked Transformer encoders should adopt last-token pooling rather than global average pooling (GAP), avoiding dilution from unevenly distributed within-window context. We implement both principles in a Causal Transformer with multi-task learning (MTL) on the AI4I 2020 dataset (10,000 samples), jointly performing fault classification and wear regression. Across the baseline and ablation configurations, the model achieves AUROC above 0.90 for Tool Wear Failure and Overstrain Failure and wear-regression R² between 0.72 and 0.84 — a level the final configuration further improves. A window-length sweep (L=10–100) shows that GAP-pooled causal models degrade sharply as windows lengthen (R² falls to 0.503 at L=100), while last-token pooling recovers R² to 0.862, nearly matching bidirectional attention. An expanded cross-validation confirms that the final configuration's wear-regression R² significantly exceeds both baselines, with TWF AUROC also significantly improved over the baseline. We further find no significant MTL synergy even under dynamic uncertainty weighting, a result we report transparently.

## 2. Data

UCI Machine Learning Repository — **AI4I 2020 Predictive Maintenance Dataset**: 10,000 synthetic CNC machine records with five documented failure modes (TWF, HDF, PWF, OSF, RNF).

Details, sequence-modeling setup, and citation: [`data/README_data.md`](data/README_data.md).

## 3. Method

### Design Principle 1 — window alignment (`src/data_windows.py::make_sliding_windows`)

Tool wear is both an input feature and part of the physical definition of two failure modes (TWF, OSF). A sliding window covering `[t-49 ... t]` with a label read from the window's own last step `t` would let a model copy the label directly from its input — training looks completely normal (the loss curve drops smoothly), but the model learns nothing about degradation trends. Instead, every window's label is read from `t+1`, a time step that never appears anywhere inside the window, which structurally prevents this leakage regardless of model architecture.

### Design Principle 2 — last-token pooling (`src/model.py::CausalTransformerMTL`)

Under a causal mask, position `i` (0-indexed) can attend to only `i+1` of the `L` window positions — the earliest positions see almost no context. Averaging all positions together (GAP) mixes these low-context representations into the pooled vector; last-token pooling instead reads only the final position, which — under a causal mask — has always seen the *entire* window. The `pooling` argument on `CausalTransformerMTL` (`'gap'` or `'last'`) makes this a controlled hyperparameter rather than a hardcoded choice, so it can be ablated directly.

### Model architecture

Causal (or bidirectional, for ablation) Transformer encoder — 2 layers, 4 heads, `d_model=64`, `dim_feedforward=256` — shared between two heads: a `Dropout + Linear` fault-classification head (5 labels, Focal Loss γ=2 for severe class imbalance) and a `Linear` wear-regression head (MSE loss). Joint loss `L = α·L_fault + β·L_wear` (α=0.7, β=0.3 fixed, or Kendall et al. 2018 learned uncertainty weighting — `src/model.py::CausalTransformerMTLDynamic`). Adam (lr=0.001), cosine-annealing LR, Xavier init, early stopping.

### Evaluation protocol

Blocked time-series cross-validation (`src/data_windows.py::build_fold_data`): data is split into `k` **consecutive** blocks; each fold uses one block as test, the next as validation, and the rest as training, with per-fold standardization fit only on that fold's training blocks. A paired t-test (`src/train_eval.py::paired_test`) compares configurations across folds.

## 4. Results

**These are numbers this repository's code actually produced when run** (8-fold blocked CV; 1 seed/fold for the two ablations, 3 seeds/fold — averaged per fold before the t-test — for the final-configuration check). The paper's own tables use a 5-fold or 8-fold protocol with up to 4 seeds/fold; per the project scope, this repo does not need to reproduce the paper's exact published numbers, only to run correctly end-to-end — and the qualitative conclusions below match the paper's throughout.

### Ablation 1 — causal mask vs. bidirectional attention (GAP pooling fixed)

| Metric | Causal+GAP | Bidirectional+GAP | p-value |
|---|---|---|---|
| TWF AUROC | 0.913 | 0.960 | 0.0005 |
| OSF AUROC | 0.905 | 0.932 | 0.0480 |
| Wear R² | 0.740 | 0.843 | 0.0011 |

Bidirectional attention wins on every metric — the same counter-intuitive direction the paper reports, motivating the mechanistic investigation below.

### Ablation 2 — MTL synergy

| Comparison | Metric | Result | p-value |
|---|---|---|---|
| Joint (fixed) vs. single-task classification | TWF AUROC | 0.960 vs 0.958 | 0.31 (not significant) |
| Joint (fixed) vs. joint (dynamic weighting) | TWF AUROC | 0.960 vs 0.959 | 0.80 (not significant) |
| Joint (fixed) vs. single-task regression | Wear R² | 0.875 vs 0.871 | 0.37 (not significant) |

No significant MTL synergy detected — consistent with the paper's transparent negative result.

### Mechanistic analysis — window-length sweep (`results/window_length_sweep.csv`)

| L | Causal+GAP wear R² | Bidirectional+GAP wear R² |
|---|---|---|
| 10 | 0.841 | 0.841 |
| 30 | 0.774 | 0.840 |
| 50 | 0.740 | 0.843 |
| 100 | 0.565 | 0.790 |

Causal+GAP degrades sharply as the window lengthens; bidirectional+GAP stays roughly flat — exactly the pattern Design Principle 2 predicts.

### Pooling ablation at L=50 (`results/pooling_ablation.csv`)

| Pooling | TWF AUROC | OSF AUROC | Wear R² |
|---|---|---|---|
| GAP | 0.921 | 0.891 | 0.746 |
| **Last-token** | **0.959** | **0.931** | **0.859** |

Switching only the pooling strategy closes nearly the entire gap to bidirectional attention (0.843 wear R² at L=50).

### Final configuration vs. baseline and bidirectional attention (8-fold × 3 seeds/fold)

| Metric | Causal+GAP (baseline) | **Causal+Last (final)** | Bidirectional+GAP | p (vs. baseline) | p (vs. bidirectional) |
|---|---|---|---|---|---|
| TWF AUROC | 0.922 | **0.955** | 0.962 | 0.0085 | 0.29 (not significant) |
| OSF AUROC | 0.903 | **0.930** | 0.929 | 0.0062 | 0.81 (not significant) |
| Wear R² | 0.755 | **0.873** | 0.851 | <0.0001 | 0.0001 |

The final configuration (causal mask + last-token pooling) significantly beats the original baseline on every metric, is statistically indistinguishable from bidirectional attention on both classification metrics, and *significantly exceeds* bidirectional attention on wear regression — reproducing the paper's core claim that causal masking's physical-consistency benefits don't have to cost any performance once paired with the right pooling strategy.

Figures: `figures/fig1_causal_mask_ablation.png`, `fig2_mtl_ablation.png`, `fig3_window_length_sweep.png`, `fig4_pooling_ablation.png`, `fig5_final_configuration.png`.

## 5. File Structure

```
causal-transformer-cnc-design-principles/
├─ data/
│  ├─ ai4i2020.csv                    Raw AI4I 2020 dataset
│  └─ README_data.md
├─ src/
│  ├─ data_windows.py                 t+1-aligned windowing, blocked CV folds, tabular summary
│  ├─ model.py                        CausalTransformerMTL, FocalLoss, dynamic-weighting variant
│  ├─ train_eval.py                   Shared train/eval loop + paired t-test
│  ├─ run_ablation_causal_mask.py     Ablation 1
│  ├─ run_ablation_mtl.py             Ablation 2
│  ├─ run_window_pooling_sweep.py     Mechanistic analysis (window length + pooling)
│  ├─ run_final_significance.py       Final configuration vs. baseline vs. bidirectional
│  └─ make_figures.py
├─ results/                           All CSVs/JSONs backing the tables above
├─ figures/                           fig1–fig5
├─ references/
└─ requirements.txt
```

## 6. How to Run

```bash
pip install -r requirements.txt

python src/run_ablation_causal_mask.py --folds 8
python src/run_ablation_mtl.py --folds 8
python src/run_window_pooling_sweep.py --window-lengths 10 30 50 100
python src/run_final_significance.py --folds 8 --seeds 42 123 2024
python src/make_figures.py
```

Each script also accepts `--max-epochs` and `--patience` (default 60/8). A CUDA GPU is used automatically if available; the full set of scripts above takes about 20 minutes on a single consumer GPU.

## 7. References

See [`references/README.md`](references/README.md). Full-text PDFs of third-party papers are not redistributed in this repository.

## 8. License

Code and documentation authored for this project (`src/`, this README) are released under the MIT License — see [`LICENSE`](LICENSE).

The following are **not** covered by that license and remain under their own terms:

- **Dataset** (`data/`): UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset, CC BY 4.0 — cite Stephan & Matzka (2020), see `data/README_data.md`.
- **References** (`references/`): copyright of the original authors/publishers — see `references/README.md`.
