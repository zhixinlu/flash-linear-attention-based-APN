#!/usr/bin/env python3
"""
Fetch experiment data from W&B and export CSV files for the LaTeX report.

Usage:
    conda run -n fla_apn python experiments/report/fetch_wandb_data.py

Output:
    experiments/report_data/<run_name>.csv          — per-epoch metrics
    experiments/report_data/<run_name>_scalars.csv   — APN eta/lam dynamics
    experiments/report_data/summary.csv              — one-row-per-run summary
"""

import argparse
import csv
import os

import wandb


WANDB_ENTITY = "zhixin-lu1988-allen-institute"
WANDB_PROJECT = "seq-cifar10-apn"
OUTDIR = os.path.join(os.path.dirname(__file__), "..", "report_data")

# Columns to export for per-epoch history
HISTORY_KEYS = [
    "epoch", "test/acc", "train/acc", "test/loss", "train/loss",
    "lr", "epoch_time_s", "peak_gpu_mb",
]

# APN scalar keys (layer 0 and last layer)
SCALAR_KEYS_TEMPLATE = [
    "epoch",
    "apn/eta_layer0", "apn/lam_layer0",
    "apn/eta_layer{last}", "apn/lam_layer{last}",
]

# Summary table columns
SUMMARY_FIELDS = [
    "name", "state", "model", "d_hidden", "n_layers", "n_heads",
    "use_ffn", "ffn_mult", "apn_activation", "n_params", "n_trainable",
    "epochs_done", "epochs_total", "warmup_epochs",
    "best_test_acc", "final_train_acc", "final_test_loss",
    "epoch_time_s", "peak_gpu_mb", "lr",
]


def fetch_all(entity: str, project: str, outdir: str, only_finished: bool = False):
    os.makedirs(outdir, exist_ok=True)
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs in {entity}/{project}\n")

    summary_rows = []

    for r in runs:
        cfg = r.config
        s = r.summary
        name = r.name
        state = r.state

        if only_finished and state != "finished":
            print(f"  SKIP {name} (state={state})")
            continue

        model = cfg.get("model", "?")
        n_layers = cfg.get("n_layers", 10)

        # --- Per-epoch history CSV ---
        hist = list(r.scan_history(keys=HISTORY_KEYS))
        if hist:
            fname = os.path.join(outdir, f"{name}.csv")
            with open(fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=HISTORY_KEYS)
                w.writeheader()
                for row in hist:
                    w.writerow({k: row.get(k, "") for k in HISTORY_KEYS})
            print(f"  {fname} ({len(hist)} epochs)")
        else:
            print(f"  {name}: no history")

        # --- APN scalar dynamics CSV ---
        if model == "apn":
            last = n_layers - 1
            scalar_keys = [k.replace("{last}", str(last)) for k in SCALAR_KEYS_TEMPLATE]
            scalar_hist = list(r.scan_history(keys=scalar_keys))
            if scalar_hist and scalar_keys[1] in scalar_hist[0]:
                fname = os.path.join(outdir, f"{name}_scalars.csv")
                with open(fname, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=scalar_keys)
                    w.writeheader()
                    for row in scalar_hist:
                        w.writerow({k: row.get(k, "") for k in scalar_keys})
                print(f"  {fname} ({len(scalar_hist)} epochs, scalars)")

        # --- Summary row ---
        summary_rows.append({
            "name": name,
            "state": state,
            "model": model,
            "d_hidden": cfg.get("d_hidden", ""),
            "n_layers": n_layers,
            "n_heads": cfg.get("n_heads", 1),
            "use_ffn": cfg.get("use_ffn", False),
            "ffn_mult": cfg.get("ffn_mult", 4),
            "apn_activation": cfg.get("apn_activation", "n/a"),
            "n_params": cfg.get("n_params", ""),
            "n_trainable": cfg.get("n_trainable", ""),
            "epochs_done": s.get("epoch", ""),
            "epochs_total": cfg.get("epochs", ""),
            "warmup_epochs": cfg.get("warmup_epochs", 0),
            "best_test_acc": s.get("test/acc", ""),
            "final_train_acc": s.get("train/acc", ""),
            "final_test_loss": s.get("test/loss", ""),
            "epoch_time_s": s.get("epoch_time_s", ""),
            "peak_gpu_mb": s.get("peak_gpu_mb", ""),
            "lr": cfg.get("lr", ""),
        })

    # --- Write summary CSV ---
    summary_path = os.path.join(outdir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)
    print(f"\n  Summary: {summary_path} ({len(summary_rows)} runs)")

    # --- Print summary table ---
    print(f"\n{'='*120}")
    print(f"{'Name':<55} {'State':<10} {'Model':<12} {'D':>4} {'Params':>8} {'Ep':>7} {'TestAcc':>8} {'Time/ep':>8} {'GPU MB':>8}")
    print(f"{'='*120}")
    for row in summary_rows:
        ep_str = f"{row['epochs_done']}/{row['epochs_total']}"
        acc = row["best_test_acc"]
        acc_str = f"{acc:.2f}%" if isinstance(acc, (int, float)) else str(acc)
        t = row["epoch_time_s"]
        t_str = f"{t:.1f}s" if isinstance(t, (int, float)) else str(t)
        gpu = row["peak_gpu_mb"]
        gpu_str = f"{gpu:.0f}" if isinstance(gpu, (int, float)) else str(gpu)
        print(f"{row['name']:<55} {row['state']:<10} {row['model']:<12} {row['d_hidden']:>4} {row['n_params']:>8} {ep_str:>7} {acc_str:>8} {t_str:>8} {gpu_str:>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch W&B data for LaTeX report")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--project", default=WANDB_PROJECT)
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--only-finished", action="store_true",
                        help="Skip runs that are not in 'finished' state")
    args = parser.parse_args()
    fetch_all(args.entity, args.project, args.outdir, args.only_finished)
