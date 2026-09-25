"""Ablation 2: does joint multi-task learning (classification + regression)
actually help versus training each head alone, and does replacing the fixed
loss weights (alpha=0.7, beta=0.3) with Kendall et al. (2018) learned
uncertainty weighting change the answer? Blocked time-series CV.
"""

import argparse
import json
import os

import pandas as pd
import torch

from data_windows import build_fold_data, load_sequence_frame, make_blocks
from model import CausalTransformerMTL, CausalTransformerMTLDynamic, init_weights_xavier
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

        m_joint = CausalTransformerMTL(input_dim=input_dim, use_causal_mask=True, pooling="last")
        m_joint.apply(init_weights_xavier); m_joint.to(device)
        m_joint = train_model(m_joint, fold_data["train_loader"], fold_data["val_loader"], device,
                               alpha=0.7, beta=0.3, seed=seed, max_epochs=max_epochs, patience=patience)
        res_joint = evaluate_model(m_joint, fold_data["test_loader"], fold_data["yf_test"], fold_data["yw_test"],
                                    fold_data["w_mean"], fold_data["w_std"], device)

        m_fault = CausalTransformerMTL(input_dim=input_dim, use_causal_mask=True, pooling="last")
        m_fault.apply(init_weights_xavier); m_fault.to(device)
        m_fault = train_model(m_fault, fold_data["train_loader"], fold_data["val_loader"], device,
                               alpha=1.0, beta=0.0, seed=seed, max_epochs=max_epochs, patience=patience)
        res_fault = evaluate_model(m_fault, fold_data["test_loader"], fold_data["yf_test"], fold_data["yw_test"],
                                    fold_data["w_mean"], fold_data["w_std"], device, eval_wear=False)

        m_wear = CausalTransformerMTL(input_dim=input_dim, use_causal_mask=True, pooling="last")
        m_wear.apply(init_weights_xavier); m_wear.to(device)
        m_wear = train_model(m_wear, fold_data["train_loader"], fold_data["val_loader"], device,
                              alpha=0.0, beta=1.0, seed=seed, max_epochs=max_epochs, patience=patience)
        res_wear = evaluate_model(m_wear, fold_data["test_loader"], fold_data["yf_test"], fold_data["yw_test"],
                                   fold_data["w_mean"], fold_data["w_std"], device, eval_fault=False)

        m_dyn = CausalTransformerMTLDynamic(input_dim=input_dim, use_causal_mask=True, pooling="last")
        m_dyn.apply(init_weights_xavier); m_dyn.to(device)
        m_dyn = train_model(m_dyn, fold_data["train_loader"], fold_data["val_loader"], device,
                             use_dynamic_weighting=True, seed=seed, max_epochs=max_epochs, patience=patience)
        res_dyn = evaluate_model(m_dyn, fold_data["test_loader"], fold_data["yf_test"], fold_data["yw_test"],
                                  fold_data["w_mean"], fold_data["w_std"], device)

        rows.append({"fold": fold, "config": "joint_fixed", "auroc_TWF": res_joint["auroc"]["TWF"],
                     "auroc_OSF": res_joint["auroc"]["OSF"], "wear_r2": res_joint["wear"]["r2"]})
        rows.append({"fold": fold, "config": "single_task_fault", "auroc_TWF": res_fault["auroc"]["TWF"],
                     "auroc_OSF": res_fault["auroc"]["OSF"], "wear_r2": None})
        rows.append({"fold": fold, "config": "single_task_wear", "auroc_TWF": None,
                     "auroc_OSF": None, "wear_r2": res_wear["wear"]["r2"]})
        rows.append({"fold": fold, "config": "joint_dynamic", "auroc_TWF": res_dyn["auroc"]["TWF"],
                     "auroc_OSF": res_dyn["auroc"]["OSF"], "wear_r2": res_dyn["wear"]["r2"],
                     "learned_sigma2_fault": float(torch.exp(m_dyn.log_var_fault).item()),
                     "learned_sigma2_wear": float(torch.exp(m_dyn.log_var_wear).item())})
        print(f"  joint_fixed TWF={res_joint['auroc']['TWF']:.4f}  single_task_fault TWF={res_fault['auroc']['TWF']:.4f}  "
              f"joint_dynamic TWF={res_dyn['auroc']['TWF']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation2_mtl.csv"), index=False)

    summary = {}
    print(f"\n=== {n_blocks}-fold blocked CV, seed={seed}: MTL ablation ===")
    for metric, a, b in [("auroc_TWF", "joint_fixed", "single_task_fault"),
                          ("auroc_TWF", "joint_fixed", "joint_dynamic"),
                          ("wear_r2", "joint_fixed", "single_task_wear")]:
        ma, mb, t, p = paired_test(df, a, b, metric)
        key = f"{a}_vs_{b}_{metric}"
        summary[key] = {a: ma, b: mb, "t": t, "p": p}
        print(f"{key}: {a}={ma:.4f}, {b}={mb:.4f}, t={t:.3f}, p={p:.4f}")

    with open(os.path.join(RESULTS_DIR, "ablation2_significance.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    main(args.folds, args.seed, args.max_epochs, args.patience)
