<a id="zh"></a>

**中文** | [English](#english)

# 從資料洩漏到架構最佳化：因果 Transformer 於 CNC 預測性維護的設計原則與實證研究

本專案為一篇研討會論文的資料處理、模型架構、消融實驗與統計驗證程式，論文針對因果注意力序列模型在預測性維護的應用，提出兩項可遷移的設計原則。

## 一、論文與研討會資料

| 項目 | 內容 |
|---|---|
| 論文題目 | From Data Leakage to Architecture Optimization: Design Principles and Empirical Study of Causal Transformers for CNC Predictive Maintenance（從資料洩漏到架構最佳化：因果 Transformer 於 CNC 預測性維護的設計原則與實證研究，論文以英文撰寫） |
| 作者 | 周理陽¹*（Li Yang Chou）、Peng Kai Wang² |
| 單位 | ¹國立中央大學機械工程學系；²國立臺灣科技大學應用科技學士學位學程 |
| 關鍵字 | 預測性維護、因果 Transformer、多任務學習、消融實驗、CNC 工具機 |

| 研討會 | 內容 |
|---|---|
| 全名 | 中華民國品質學會第62屆年會暨2026國際品質管理研討會（Chinese Society for Quality 62nd Annual Meeting & 2026 International Symposium on Quality Management, ISQM 2026） |
| 主辦 | 中華民國品質學會（CSQ）、元智大學工業工程與管理學系 |
| 日期 | 2026 年 11 月 7 日 |
| 地點 | 元智大學有庠廳（桃園市） |
| 官方網站 | https://sites.google.com/view/isqm2026 |
| 狀態 | 已錄取 |

### 論文摘要（中文為譯文，原文見下方英文部分）

有鑑於將因果注意力序列模型應用於預測性維護時兩個常被忽略的設計陷阱，本研究以 CNC 刀具磨損預測為案例，提出並驗證兩項對應的設計原則。第一，**窗口對齊原則**：當輸入特徵與預測目標具自我參照關係時，應將目標對齊到下一時間點（t+1），而非窗口本身的最末時間點，從結構上避免資料洩漏。第二，**池化策略原則**：因果遮罩的 Transformer 編碼器應採用最末時間點池化（last-token pooling），而非全局平均池化（GAP），以避免窗口內上下文分布不均所造成的稀釋。我們將兩項原則實作於以 AI4I 2020 資料集（10,000 筆）訓練的因果 Transformer 多任務學習（MTL）模型，同時進行故障分類與磨損迴歸。在基準與各消融配置下，模型於刀具磨損故障與過度應變故障皆達 AUROC 0.90 以上，磨損迴歸 R² 介於 0.72 至 0.84，而最終配置進一步提升此水準。窗口長度掃描（L=10–100）顯示 GAP 池化的因果模型隨窗口增長急遽劣化（L=100 時 R² 降至 0.503），而最末時間點池化將 R² 回復至 0.862，幾乎追平雙向注意力。擴大後的交叉驗證證實最終配置的磨損迴歸 R² 顯著超越兩個基準，且 TWF AUROC 相較基準也顯著提升。我們並發現，即使採用動態不確定性加權，仍無顯著的 MTL 協同效應，此結果如實揭露。

## 二、資料

UCI Machine Learning Repository 的 **AI4I 2020 預測性維護資料集**：10,000 筆合成 CNC 機台紀錄，含五種故障模式（TWF、HDF、PWF、OSF、RNF）。

欄位定義、序列建模設定與引用方式見 [`data/README_data.md`](data/README_data.md)。

## 三、方法

### 設計原則一：窗口對齊（`src/data_windows.py::make_sliding_windows`）

刀具磨損既是輸入特徵，又是兩種故障模式（TWF、OSF）的物理定義之一。若窗口涵蓋 `[t-49 ... t]`，卻從窗口本身的最末時間點 `t` 取標籤，模型可以直接從輸入抄答案——訓練看起來完全正常（損失曲線平滑下降），但模型什麼退化趨勢都沒學到。因此每個窗口的標籤都取自 `t+1`，這個時間點完全不在窗口內，無論模型架構為何，都從結構上避免了這種洩漏。

### 設計原則二：最末時間點池化（`src/model.py::CausalTransformerMTL`）

在因果遮罩下，位置 `i`（從 0 起算）只能關注 `L` 個窗口位置中的 `i+1` 個——窗口前段的位置幾乎看不到上下文。全局平均（GAP）會把這些低上下文的表徵一併平均進池化向量；最末時間點池化則只讀最後一個位置，而在因果遮罩下，該位置**總是**看過整個窗口。`CausalTransformerMTL` 的 `pooling` 參數（`'gap'` 或 `'last'`）把它做成可控的超參數而非寫死的選擇，因此可以直接做消融。

### 模型架構

因果（或消融用的雙向）Transformer 編碼器——2 層、4 個注意力頭、`d_model=64`、`dim_feedforward=256`——由兩個輸出頭共用：`Dropout + Linear` 的故障分類頭（5 個標籤，以 Focal Loss γ=2 處理嚴重類別不平衡），以及 `Linear` 的磨損迴歸頭（MSE 損失）。聯合損失為 `L = α·L_fault + β·L_wear`（固定 α=0.7、β=0.3，或 Kendall et al. 2018 的可學習不確定性加權——`src/model.py::CausalTransformerMTLDynamic`）。優化器 Adam（lr=0.001），餘弦退火學習率，Xavier 初始化，早停。

### 評估流程

時間區塊交叉驗證（`src/data_windows.py::build_fold_data`）：資料切成 `k` 個**連續**區塊；每一折以一個區塊為測試集、下一個為驗證集、其餘為訓練集，每折的標準化只以該折訓練區塊配適。以成對 t 檢定（`src/train_eval.py::paired_test`）跨折比較各配置。

## 四、執行結果

**以下為本 repo 程式碼實際執行所產出的數字**（8 折時間區塊交叉驗證；兩項消融每折 1 個種子；最終配置檢驗每折 3 個種子，先在折內平均再做 t 檢定）。論文本身的表格採用 5 折或 8 折、最多每折 4 個種子；依專案範圍，本 repo 不需重現論文的確切數字，只需正確地端到端執行——而下列質性結論與論文全程一致。

### 消融一：因果遮罩 vs. 雙向注意力（固定 GAP 池化）

| 指標 | 因果+GAP | 雙向+GAP | p 值 |
|---|---|---|---|
| TWF AUROC | 0.913 | 0.960 | 0.0005 |
| OSF AUROC | 0.905 | 0.932 | 0.0480 |
| 磨損 R² | 0.740 | 0.843 | 0.0011 |

雙向注意力在每項指標都勝出——與論文報告的違反直覺方向相同，也因此引出下面的機制性研究。

### 消融二：MTL 協同效應

| 比較 | 指標 | 結果 | p 值 |
|---|---|---|---|
| 聯合（固定權重）vs. 單任務分類 | TWF AUROC | 0.960 vs 0.958 | 0.31（不顯著） |
| 聯合（固定權重）vs. 聯合（動態加權） | TWF AUROC | 0.960 vs 0.959 | 0.80（不顯著） |
| 聯合（固定權重）vs. 單任務迴歸 | 磨損 R² | 0.875 vs 0.871 | 0.37（不顯著） |

未偵測到顯著的 MTL 協同效應——與論文如實揭露的負面結果一致。

### 機制性分析：窗口長度掃描（`results/window_length_sweep.csv`）

| L | 因果+GAP 磨損 R² | 雙向+GAP 磨損 R² |
|---|---|---|
| 10 | 0.841 | 0.841 |
| 30 | 0.774 | 0.840 |
| 50 | 0.740 | 0.843 |
| 100 | 0.565 | 0.790 |

因果+GAP 隨窗口增長急遽劣化，雙向+GAP 則大致持平——正是設計原則二所預測的樣貌。

### 池化消融（L=50，`results/pooling_ablation.csv`）

| 池化 | TWF AUROC | OSF AUROC | 磨損 R² |
|---|---|---|---|
| GAP | 0.921 | 0.891 | 0.746 |
| **最末時間點** | **0.959** | **0.931** | **0.859** |

只改變池化策略，就幾乎完全補平了與雙向注意力（L=50 時磨損 R² 為 0.843）的差距。

### 最終配置 vs. 基準與雙向注意力（8 折 × 每折 3 個種子）

| 指標 | 因果+GAP（基準） | **因果+最末時間點（最終）** | 雙向+GAP | p（對基準） | p（對雙向） |
|---|---|---|---|---|---|
| TWF AUROC | 0.922 | **0.955** | 0.962 | 0.0085 | 0.29（不顯著） |
| OSF AUROC | 0.903 | **0.930** | 0.929 | 0.0062 | 0.81（不顯著） |
| 磨損 R² | 0.755 | **0.873** | 0.851 | <0.0001 | 0.0001 |

最終配置（因果遮罩 + 最末時間點池化）在每項指標都顯著優於原始基準，在兩項分類指標上與雙向注意力統計上無差異，並且在磨損迴歸上**顯著超越**雙向注意力——重現了論文的核心主張：只要搭配正確的池化策略，因果遮罩帶來的物理一致性不必以犧牲效能為代價。

圖表：`figures/fig1_causal_mask_ablation.png`、`fig2_mtl_ablation.png`、`fig3_window_length_sweep.png`、`fig4_pooling_ablation.png`、`fig5_final_configuration.png`。

## 五、檔案結構

```
causal-transformer-cnc-design-principles/
├─ data/
│  ├─ ai4i2020.csv                    AI4I 2020 原始資料
│  └─ README_data.md
├─ src/
│  ├─ data_windows.py                 t+1 對齊窗口、時間區塊 CV 折、表格摘要
│  ├─ model.py                        CausalTransformerMTL、FocalLoss、動態加權變體
│  ├─ train_eval.py                   共用訓練／評估迴圈 + 成對 t 檢定
│  ├─ run_ablation_causal_mask.py     消融一
│  ├─ run_ablation_mtl.py             消融二
│  ├─ run_window_pooling_sweep.py     機制性分析（窗口長度 + 池化）
│  ├─ run_final_significance.py       最終配置 vs. 基準 vs. 雙向
│  └─ make_figures.py
├─ results/                           支撐上述表格的所有 CSV／JSON
├─ figures/                           fig1–fig5
├─ references/
└─ requirements.txt
```

## 六、如何執行

```bash
pip install -r requirements.txt

python src/run_ablation_causal_mask.py --folds 8
python src/run_ablation_mtl.py --folds 8
python src/run_window_pooling_sweep.py --window-lengths 10 30 50 100
python src/run_final_significance.py --folds 8 --seeds 42 123 2024
python src/make_figures.py
```

各腳本另可用 `--max-epochs` 與 `--patience`（預設 60／8）。若有 CUDA GPU 會自動使用；以上全部腳本在一張一般消費級 GPU 上約需 20 分鐘。

## 七、參考文獻

見 [`references/README.md`](references/README.md)。第三方論文全文不隨本 repo 散布。

## 八、授權

本專案自行撰寫的程式碼（`src/`）與文件以 MIT License 釋出，詳見 [`LICENSE`](LICENSE)。

以下內容不在本授權範圍內，各自沿用原本的條款：

- **資料集**（`data/`）：UCI Machine Learning Repository 的 AI4I 2020 預測性維護資料集，CC BY 4.0，引用 Stephan & Matzka (2020)，見 `data/README_data.md`。
- **參考文獻**（`references/`）：著作權歸各作者與出版方所有，見 `references/README.md`。

---

<a id="english"></a>

[中文](#zh) | **English**

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
