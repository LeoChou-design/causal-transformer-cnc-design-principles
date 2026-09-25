"""Final configuration check: causal+last-token (the configuration Design
Principles 1+2 recommend) vs. the original causal+GAP baseline vs.
bidirectional+GAP, under blocked time-series CV with multiple seeds per
fold (fold-level results are averaged across seeds first, then compared
across folds with a paired t-test) — reduces the risk of a single unlucky
random initialization being mistaken for a fold effect.
"""

import argparse
import json
import os

import pandas as pd
import torch

from data_windows import build_fold_data, load_sequence_frame, make_blocks
from model import CausalTransformerMTL, init_weights_xavier
from train_eval import evaluate_model, paired_test, train_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

CONFIGS = [
    ("causal+gap (original baseline)", True, "gap"),
    ("causal+last (final configuration)", True, "last"),
    ("bidirectional+gap", False, "gap"),
]


def main(n_blocks, seeds, max_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_seq, feature_cols = load_sequence_frame()
    blocks = make_blocks(df_seq, n_blocks)
    input_dim = len(feature_cols)

    raw_rows = []
    for fold in range(n_blocks):
        print(f"--- Fold {fold + 1}/{n_blocks} ---")
        fold_data = build_fold_data(fold, blocks, n_blocks, feature_cols)
        for name, use_mask, pooling in CONFIGS:
            for seed in seeds:
                model = CausalTransformerMTL(input_dim=input_dim, use_causal_mask=use_mask, pooling=pooling)
                model.apply(init_weights_xavier)
                model.to(device)
                model = train_model(model, fold_data["train_loader"], fold_data["val_loader"], device,
                                     alpha=0.7, beta=0.3, seed=seed, max_epochs=max_epochs, patience=patience)
                res = evaluate_model(model, fold_data["test_loader"], fold_data["yf_test"], fold_data["yw_test"],
                                      fold_data["w_mean"], fold_data["w_std"], device)
                raw_rows.append({"fold": fold, "config": name, "seed": seed,
                                  "auroc_TWF": res["auroc"]["TWF"], "auroc_OSF": res["auroc"]["OSF"],
                                  "wear_r2": res["wear"]["r2"]})
        print(f"  fold {fold + 1} done ({len(CONFIGS)} configs x {len(seeds)} seeds)")

    raw_df = pd.DataFrame(raw_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_df.to_csv(os.path.join(RESULTS_DIR, "final_significance_raw.csv"), index=False)

    fold_avg = raw_df.groupby(["fold", "config"])[["auroc_TWF", "auroc_OSF", "wear_r2"]].mean().reset_index()
    fold_avg.to_csv(os.path.join(RESULTS_DIR, "final_significance_fold_avg.csv"), index=False)

    print(f"\n=== {n_blocks}-fold blocked CV x {len(seeds)} seeds/fold: mean +/- std ===")
    summary_table = fold_avg.groupby("config")[["auroc_TWF", "auroc_OSF", "wear_r2"]].agg(["mean", "std"])
    print(summary_table)

    sig = {}
    baseline, final_cfg, bidir = [c[0] for c in CONFIGS]
    for metric in ["auroc_TWF", "auroc_OSF", "wear_r2"]:
        for a, b in [(baseline, final_cfg), (bidir, final_cfg)]:
            ma, mb, t, p = paired_test(fold_avg, a, b, metric)
            key = f"{metric}: {a} vs {b}"
            sig[key] = {"mean_a": ma, "mean_b": mb, "t": t, "p": p}
            print(f"{key}: {a}={ma:.4f}, {b}={mb:.4f}, t={t:.3f}, p={p:.4f}")

    with open(os.path.join(RESULTS_DIR, "final_significance.json"), "w") as f:
        json.dump(sig, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(args.folds, args.seeds, args.max_epochs, args.patience)
