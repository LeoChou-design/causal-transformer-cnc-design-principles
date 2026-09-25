"""Ablation 1: does the causal mask help or hurt, holding pooling fixed at
GAP? Blocked time-series CV, paired t-test between causal and bidirectional
attention. This is the ablation that originally motivated Design Principle 2
(see README) — a naive reading of the result below argues bidirectional
attention is better, until Unit/script 3 shows the effect disappears once
pooling is switched to last-token.
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


def main(n_blocks, seed, max_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_seq, feature_cols = load_sequence_frame()
    blocks = make_blocks(df_seq, n_blocks)
    input_dim = len(feature_cols)

    rows = []
    for fold in range(n_blocks):
        print(f"--- Fold {fold + 1}/{n_blocks} ---")
        fold_data = build_fold_data(fold, blocks, n_blocks, feature_cols)

        for use_mask, name in [(True, "causal+gap"), (False, "bidirectional+gap")]:
            model = CausalTransformerMTL(input_dim=input_dim, use_causal_mask=use_mask, pooling="gap")
            model.apply(init_weights_xavier)
            model.to(device)
            model = train_model(model, fold_data["train_loader"], fold_data["val_loader"],
                                 device, alpha=0.7, beta=0.3, seed=seed,
                                 max_epochs=max_epochs, patience=patience)
            res = evaluate_model(model, fold_data["test_loader"], fold_data["yf_test"], fold_data["yw_test"],
                                  fold_data["w_mean"], fold_data["w_std"], device)
            row = {"fold": fold, "config": name, "wear_r2": res["wear"]["r2"]}
            row.update({f"auroc_{k}": v for k, v in res["auroc"].items()})
            rows.append(row)
            print(f"  [{name}] TWF={row.get('auroc_TWF')}, OSF={row.get('auroc_OSF')}, wear_r2={row['wear_r2']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation1_causal_mask.csv"), index=False)

    summary = {}
    print(f"\n=== {n_blocks}-fold blocked CV, seed={seed}: causal+gap vs bidirectional+gap ===")
    for metric in ["auroc_TWF", "auroc_OSF", "wear_r2"]:
        ma, mb, t, p = paired_test(df, "causal+gap", "bidirectional+gap", metric)
        summary[metric] = {"causal+gap": ma, "bidirectional+gap": mb, "t": t, "p": p}
        print(f"{metric}: causal+gap={ma:.4f}, bidirectional+gap={mb:.4f}, t={t:.3f}, p={p:.4f}")

    with open(os.path.join(RESULTS_DIR, "ablation1_significance.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(args.folds, args.seed, args.max_epochs, args.patience)
