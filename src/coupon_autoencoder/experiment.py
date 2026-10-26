"""One fixed benchmark run. Synthetic labels are used only for holdout evaluation."""

import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from .data import FEATURES, generate, write_dataset
from .model import Detector, save_bundle


def metrics(labels, scores, threshold):
    predicted = scores > threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    both_classes = len(np.unique(labels)) == 2
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "false_positive_rate": float(fp / (tn + fp)) if tn + fp else None,
        "alert_count": int(predicted.sum()), "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }


def run(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = generate(seed=seed)
    masks = {s: data.split == s for s in ("train", "validation", "test")}
    train, validation, test = (masks[s] for s in ("train", "validation", "test"))
    if data.labels[train].any() or data.labels[validation].any():
        raise ValueError("This benchmark requires clean training and calibration data")
    detector = Detector(seed=seed).fit(data.x[train], data.x[validation])
    test_scores = detector.scores(data.x[test])
    val_scores = detector.scores(data.x[validation])
    truth = data.labels[test]
    results = {name: metrics(truth, scores, detector.thresholds_[name]) for name, scores in test_scores.items()}
    auto_flags = test_scores["autoencoder"] > detector.thresholds_["autoencoder"]
    subgroups = {}
    for scenario in sorted(set(data.scenario[test])):
        m = data.scenario[test] == scenario
        subgroups[scenario] = {"count": int(m.sum()), "alert_count": int(auto_flags[m].sum()),
                              "alert_rate": float(auto_flags[m].mean())}
    write_dataset(data, output / "synthetic_features.csv")
    # A label-free file is used to demonstrate independent scoring.
    with (output / "inference_example.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(FEATURES); w.writerows(data.x[test][:15])
    save_bundle(detector, output / "model.json")
    report = {
        "seed": seed, "data_source": "synthetic_only", "n_features": len(FEATURES),
        "split_unit": "day_index", "same_employees_across_periods": True,
        "splits": {s: {"rows": int(m.sum()), "anomalies": int(data.labels[m].sum()),
                        "first_day": int(data.day[m].min()), "last_day": int(data.day[m].max())} for s, m in masks.items()},
        "threshold_method": "99th percentile of later normal validation scores, higher quantile",
        "label_use": "evaluation_only; no labels or test rows in model fitting or threshold selection",
        "autoencoder": {"layers": [12, 24, 6, 24, 12], "hidden_activation": "tanh",
                        "output_activation": "identity", "epochs": int(detector.autoencoder_.n_iter_),
                        "initial_training_loss": float(detector.autoencoder_.loss_curve_[0]),
                        "final_training_loss": float(detector.autoencoder_.loss_curve_[-1]),
                        "warnings": detector.training_warnings_},
        "normal_validation_alert_rates": {n: float((s > detector.thresholds_[n]).mean()) for n, s in val_scores.items()},
        "models": results, "autoencoder_subgroups": subgroups,
        "versions": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "matplotlib"]}},
        "dataset_sha256": hashlib.sha256((output / "synthetic_features.csv").read_bytes()).hexdigest(),
    }
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "loss_curve.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["epoch", "training_loss"])
        w.writerows(enumerate(detector.autoencoder_.loss_curve_, start=1))
    test_ids = np.flatnonzero(test)
    contributions = detector.contributions(data.x[test])
    order = np.argsort(-test_scores["autoencoder"], kind="stable")[:25]
    with (output / "review_queue.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["day", "employee_id", "score", "review_flag", "synthetic_label", "synthetic_scenario", "largest_reconstruction_residuals"])
        for j in order:
            i = test_ids[j]
            top = np.argsort(-contributions[j])[:3]
            # Residuals locate reconstruction differences, not causal explanations.
            w.writerow([int(data.day[i]), f"E{data.employee[i]:04d}", float(test_scores["autoencoder"][j]),
                        bool(auto_flags[j]), int(data.labels[i]), data.scenario[i], ";".join(FEATURES[k] for k in top)])
    plot(output, detector, test_scores, truth, report)
    lines = ["# Synthetic benchmark results", "", f"Seed: {seed}. Source: synthetic data only.",
             "Train days 0–59; calibration days 60–89; holdout days 90–119.",
             "The same synthetic employees appear in each period; unseen-employee generalization is not measured.", ""]
    for name, m in results.items():
        lines.append(f"- **{name}**: AP={m['average_precision']:.4f}; precision={m['precision']:.4f}; recall={m['recall']:.4f}; FPR={m['false_positive_rate']:.4f}; TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']}.")
    lines += ["", "AP is average precision, not trapezoidal PR-AUC. Test anomaly prevalence is 5%.",
              "No architecture or threshold selection is performed on the holdout.",
              "This is a generator-specific demonstration, not evidence of fraud detection quality at a bank.",
              "Model comparisons are reported as observed; an autoencoder need not beat simpler baselines.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def plot(output, detector, scores, truth, report):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for label, title, color in [(0, "Normal", "#247c8c"), (1, "Injected anomaly", "#d06d40")]:
        axes[0].hist(np.log10(scores["autoencoder"][truth == label] + 1e-9), bins=35, alpha=.65,
                     label=title, color=color, density=True)
    axes[0].axvline(np.log10(detector.thresholds_["autoencoder"] + 1e-9), color="#23323f", linestyle="--", label="Validation threshold")
    axes[0].set(xlabel="log10 reconstruction MSE", ylabel="Density", title="Synthetic holdout scores")
    axes[0].legend(fontsize=8)
    names = list(scores)
    positions = np.arange(len(names))
    axes[1].bar(positions - .18, [report["models"][n]["average_precision"] for n in names], .36, label="Average precision", color="#247c8c")
    axes[1].bar(positions + .18, [report["models"][n]["recall"] for n in names], .36, label="Recall at calibrated threshold", color="#d7a465")
    axes[1].set_xticks(positions, ["Autoencoder", "PCA", "Isolation\nForest", "Max-|z|\nrule"], fontsize=9)
    axes[1].set(ylim=(0, 1.08), ylabel="Score", title="Same data; same calibration target")
    axes[1].legend(fontsize=8)
    fig.suptitle("Coupon-terminal anomalies · synthetic data only", fontsize=13, fontweight="bold")
    fig.savefig(output / "benchmark.png", dpi=160)
    plt.close(fig)
