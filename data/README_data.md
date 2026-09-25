# Data: AI4I 2020 Predictive Maintenance Dataset

10,000 synthetic CNC machine records, in original time (UDI) order, with
five documented failure modes.

| File | Content | Size |
|---|---|---|
| `ai4i2020.csv` | Raw dataset as released by the UCI Machine Learning Repository | 10,000 rows × 14 columns |

## Sequence modeling setup

- **Input features (8-dim)**: 5 continuous columns (air temperature, process
  temperature, rotational speed, torque, tool wear) + 3 one-hot columns for
  machine type (L/M/H).
- **Targets**: 5 binary fault labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) +
  tool wear (continuous).
- **Window-alignment principle** (see `src/data_windows.py::make_sliding_windows`):
  a sliding window covers time steps `[t-49 ... t]` (length 50), and its
  label is read from `t+1` — not from the window's own last step. Tool wear
  is both an input feature and part of the physical definition of TWF/OSF;
  reading the label from `t` instead of `t+1` would let a model copy the
  answer directly from its input (see README §3 for the full explanation
  and a worked numeric example in `results/`).

## Source

- Dataset page: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- Citation: Stephan, M., & Matzka, S. (2020). AI4I 2020 Predictive
  Maintenance Dataset. UCI Machine Learning Repository.
  https://doi.org/10.1109/AI4I49448.2020.00023 (companion paper)
- License: CC BY 4.0 (UCI Machine Learning Repository standard license).
