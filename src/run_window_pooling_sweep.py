"""Mechanistic analysis behind Design Principle 2. Uses the single
time-ordered 80/10/10 split (not blocked CV — this is a mechanism check,
not a headline comparison).

Part A — window-length sweep: under a causal mask, position i can attend to
only i+1 of the L window positions, so the *average* available context is
(L+1)/2 regardless of L, but the *distribution* worsens as L grows (more
low-context positions get averaged into a GAP-pooled vector). Prediction:
causal+GAP performance should degrade as window length L grows; bidirectional
attention (every position sees all L positions) should stay flat.

Part B — pooling ablation at a fixed L=50: does switching from GAP to
last-token pooling close the gap between causal and bidirectional attention?
"""

import argparse
import os

import pandas as pd
import torch

from data_windows import (
    build_loaders, load_sequence_frame, make_sliding_windows, single_split,
)
from model import CausalTransformerMTL, init_weights_xavier
from train_eval import evaluate_model, train_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_one(input_dim, window_length, use_causal_mask, pooling, df_train, df_val, df_test,
            feature_cols, device, seed, max_epochs, patience):
    X_tr, yf_tr, yw_tr = make_sliding_windows(df_train, feature_cols, window_length=window_length)
    X_va, yf_va, yw_va = make_sliding_windows(df_val, feature_cols, window_length=window_length)
    X_te, yf_te, yw_te = make_sliding_windows(df_test, feature_cols, window_length=window_length)

    # Wear-target standardization uses this window length's own training
    # statistics (row count changes slightly with L).
    w_mean, w_std = yw_tr.mean(), yw_tr.std()
    train_loader = build_loaders(X_tr, yf_tr, yw_tr, w_mean, w_std, shuffle=True)
    val_loader = build_loaders(X_va, yf_va, yw_va, w_mean, w_std, shuffle=False)
    test_loader = build_loaders(X_te, yf_te, yw_te, w_mean, w_std, shuffle=False)

    model = CausalTransformerMTL(input_dim=input_dim, max_len=window_length,
                                  use_causal_mask=use_causal_mask, pooling=pooling)
    model.apply(init_weights_xavier)
    model.to(device)
    model = train_model(model, train_loader, val_loader, device, alpha=0.7, beta=0.3, seed=seed,
                         max_epochs=max_epochs, patience=patience)
    return evaluate_model(model, test_loader, yf_te, yw_te, w_mean, w_std, device)


def main(window_lengths, seed, max_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_seq, feature_cols = load_sequence_frame()
    df_train, df_val, df_test = single_split(df_seq)
    input_dim = len(feature_cols)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Part A: window-length sweep, GAP pooling, causal vs bidirectional
    sweep_rows = []
    for L in window_lengths:
        for use_mask, name in [(True, "causal+gap"), (False, "bidirectional+gap")]:
            res = run_one(input_dim, L, use_mask, "gap", df_train, df_val, df_test,
                           feature_cols, device, seed, max_epochs, patience)
            sweep_rows.append({"window_length": L, "config": name, "wear_r2": res["wear"]["r2"]})
            print(f"L={L}, {name}: wear_r2={res['wear']['r2']:.4f}")
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(os.path.join(RESULTS_DIR, "window_length_sweep.csv"), index=False)
    print("\nWindow-length sweep (wear R2):")
    print(sweep_df.pivot(index="window_length", columns="config", values="wear_r2"))

    # Part B: pooling ablation at L=50, causal mask fixed
    pooling_rows = []
    for pooling in ["gap", "last"]:
        res = run_one(input_dim, 50, True, pooling, df_train, df_val, df_test,
                       feature_cols, device, seed, max_epochs, patience)
        pooling_rows.append({"pooling": pooling, "auroc_TWF": res["auroc"]["TWF"],
                              "auroc_OSF": res["auroc"]["OSF"], "wear_r2": res["wear"]["r2"]})
        print(f"causal+{pooling}: TWF={res['auroc']['TWF']:.4f}, OSF={res['auroc']['OSF']:.4f}, "
              f"wear_r2={res['wear']['r2']:.4f}")
    # Also record bidirectional+GAP at L=50 as the reference point already computed above
    bidir_l50 = sweep_df[(sweep_df["window_length"] == 50) & (sweep_df["config"] == "bidirectional+gap")]
    pooling_df = pd.DataFrame(pooling_rows)
    pooling_df.to_csv(os.path.join(RESULTS_DIR, "pooling_ablation.csv"), index=False)
    print("\nPooling ablation (L=50):")
    print(pooling_df)
    if not bidir_l50.empty:
        print(f"(reference — bidirectional+gap wear_r2 at L=50: {bidir_l50['wear_r2'].iloc[0]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-lengths", type=int, nargs="+", default=[10, 30, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(args.window_lengths, args.seed, args.max_epochs, args.patience)
