"""
Step 8: Evaluation & Paper Figures
Comprehensive evaluation: metrics, statistical tests, ablation table, figures.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_auc_score,
)
from scipy.stats import chi2
from itertools import combinations

import config


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """
    McNemar's test for comparing two classifiers.
    Returns p-value.
    """
    # Contingency: where A is correct and B is wrong, and vice versa
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    # b: A correct, B wrong; c: A wrong, B correct
    b_count = np.sum(a_correct & ~b_correct)
    c_count = np.sum(~a_correct & b_correct)

    if b_count + c_count == 0:
        return 1.0

    # McNemar statistic with continuity correction
    stat = (abs(b_count - c_count) - 1) ** 2 / (b_count + c_count)
    p_value = 1 - chi2.cdf(stat, df=1)
    return p_value


def bootstrap_f1(y_true: np.ndarray, y_pred: np.ndarray,
                 n_iter: int = None, seed: int = 42) -> tuple[float, float, float]:
    """
    BCa (Bias-Corrected and Accelerated) bootstrap confidence interval for macro F1.
    Returns (mean, lower_95, upper_95).
    """
    from scipy.stats import norm as sp_norm
    n_iter = n_iter or config.BOOTSTRAP_N_ITERATIONS
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Observed statistic
    observed = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Bootstrap replicates
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        try:
            f1 = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
            scores.append(f1)
        except Exception:
            continue
    scores = np.array(scores)
    if len(scores) < 100:
        return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

    # BCa bias correction factor z0
    z0 = sp_norm.ppf(np.mean(scores < observed))
    if np.isinf(z0):
        z0 = 0.0

    # Acceleration factor via jackknife
    jackknife_scores = []
    for i in range(n):
        jk_mask = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        try:
            jk_f1 = f1_score(y_true[jk_mask], y_pred[jk_mask], average="macro", zero_division=0)
            jackknife_scores.append(jk_f1)
        except Exception:
            jackknife_scores.append(observed)
    jackknife_scores = np.array(jackknife_scores)
    jk_mean = np.mean(jackknife_scores)
    jk_diff = jk_mean - jackknife_scores
    a = np.sum(jk_diff ** 3) / (6.0 * (np.sum(jk_diff ** 2)) ** 1.5 + 1e-12)

    # Adjusted percentiles
    alpha = 0.05
    z_alpha_lo = sp_norm.ppf(alpha / 2)
    z_alpha_hi = sp_norm.ppf(1 - alpha / 2)

    def bca_percentile(z_alpha):
        num = z0 + z_alpha
        adjusted = z0 + num / (1.0 - a * num + 1e-12)
        return sp_norm.cdf(adjusted) * 100

    lo_pct = bca_percentile(z_alpha_lo)
    hi_pct = bca_percentile(z_alpha_hi)
    lo_pct = np.clip(lo_pct, 0.5, 99.5)
    hi_pct = np.clip(hi_pct, 0.5, 99.5)

    return float(np.mean(scores)), float(np.percentile(scores, lo_pct)), float(np.percentile(scores, hi_pct))


def compute_auc_from_hard_labels(y_true: np.ndarray, y_pred: np.ndarray,
                                  label_names: list) -> float:
    """
    Compute macro-averaged one-vs-rest AUC-ROC from hard label predictions.

    Since GPT returns hard labels (not probabilities), we convert predictions
    to binary indicator vectors and compute AUC per class vs rest.
    This is a point estimate on the ROC curve — equivalent to balanced accuracy
    when using hard labels. For XGBoost we use predict_proba() instead
    (see compute_auc_from_proba), which gives a true AUC.

    Note: AUC from hard labels = 0.5 * (TPR + TNR) per class, then averaged.
    This still penalises class imbalance fairly and is comparable across models.
    """
    try:
        from sklearn.preprocessing import label_binarize
        if len(label_names) == 2:
            # Binary: map to 0/1
            le_map = {v: i for i, v in enumerate(label_names)}
            yt_bin = np.array([le_map.get(v, -1) for v in y_true])
            yp_bin = np.array([le_map.get(v, -1) for v in y_pred])
            mask = (yt_bin >= 0) & (yp_bin >= 0)
            if mask.sum() < 2:
                return float("nan")
            return float(roc_auc_score(yt_bin[mask], yp_bin[mask]))
        else:
            # Multiclass OvR
            yt_bin = label_binarize(y_true, classes=label_names)
            yp_bin = label_binarize(y_pred, classes=label_names)
            # Only include classes present in y_true
            present = [i for i, cls in enumerate(label_names)
                       if cls in y_true]
            if len(present) < 2:
                return float("nan")
            return float(roc_auc_score(
                yt_bin[:, present], yp_bin[:, present],
                average="macro", multi_class="ovr",
            ))
    except Exception:
        return float("nan")


def compute_auc_from_proba(y_true: np.ndarray, y_prob: np.ndarray,
                            label_names: list) -> float:
    """
    Compute macro-averaged AUC-ROC from probability scores (XGBoost only).
    This is a true AUC using the full ROC curve, not just a point estimate.
    """
    try:
        from sklearn.preprocessing import label_binarize
        if len(label_names) == 2:
            # Binary: use probability of positive class (index 1)
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            yt_bin = label_binarize(y_true, classes=label_names)
            present = [i for i, cls in enumerate(label_names)
                       if cls in y_true]
            if len(present) < 2:
                return float("nan")
            return float(roc_auc_score(
                yt_bin[:, present], y_prob[:, present],
                average="macro", multi_class="ovr",
            ))
    except Exception:
        return float("nan")


def compute_metrics(y_true, y_pred, label_names) -> dict:
    """Compute all evaluation metrics."""
    mask = pd.notna(y_true) & pd.notna(y_pred)
    yt = np.array(y_true[mask])
    yp = np.array(y_pred[mask])

    if len(yt) == 0:
        return {"n_valid": 0}

    acc = accuracy_score(yt, yp)
    f1_macro = f1_score(yt, yp, average="macro", zero_division=0)
    f1_weighted = f1_score(yt, yp, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(yt, yp)

    prec, rec, f1_per, support = precision_recall_fscore_support(
        yt, yp, labels=label_names, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(label_names):
        per_class[name] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1_per[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(yt, yp, labels=label_names)

    # Bootstrap CI
    f1_mean, f1_lo, f1_hi = bootstrap_f1(yt, yp, config.BOOTSTRAP_N_ITERATIONS)

    # AUC-ROC from hard labels (works for both GPT and XGBoost)
    # For XGBoost with probabilities, call compute_auc_from_proba separately
    auc_hard = compute_auc_from_hard_labels(yt, yp, label_names)

    return {
        "n_valid": int(np.sum(mask)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_macro_ci_lower": f1_lo,
        "f1_macro_ci_upper": f1_hi,
        "cohen_kappa": float(kappa),
        "auc_roc": float(auc_hard) if not (isinstance(auc_hard, float) and auc_hard != auc_hard) else None,
        "auc_source": "hard_labels",
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }



# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_majority_baseline(y_true: np.ndarray, label_names: list) -> dict:
    """Majority-class baseline: always predict the most frequent class."""
    from collections import Counter
    counts = Counter(y_true)
    majority_class = counts.most_common(1)[0][0]
    y_pred = np.array([majority_class] * len(y_true))
    metrics = compute_metrics(pd.Series(y_true), pd.Series(y_pred), label_names)
    metrics["majority_class"] = majority_class
    metrics["majority_class_freq"] = round(counts[majority_class] / len(y_true), 4)
    metrics["cost_usd"] = 0.0
    metrics["null_rate"] = 0.0
    return metrics


def compute_random_baseline(y_true: np.ndarray, label_names: list,
                             n_runs: int = None, seed: int = 42) -> dict:
    """Random baseline: predict uniformly at random from label_names, n_runs times."""
    n_runs = n_runs or getattr(config, "RANDOM_BASELINE_RUNS", 1000)
    rng = np.random.RandomState(seed)
    f1_scores, acc_scores = [], []
    for _ in range(n_runs):
        y_pred = rng.choice(label_names, size=len(y_true))
        f1_scores.append(
            f1_score(y_true, y_pred, average="macro",
                     labels=label_names, zero_division=0)
        )
        acc_scores.append(accuracy_score(y_true, y_pred))
    return {
        "n_valid": len(y_true),
        "accuracy": float(np.mean(acc_scores)),
        "accuracy_std": float(np.std(acc_scores)),
        "f1_macro": float(np.mean(f1_scores)),
        "f1_macro_std": float(np.std(f1_scores)),
        "f1_macro_ci_lower": float(np.percentile(f1_scores, 2.5)),
        "f1_macro_ci_upper": float(np.percentile(f1_scores, 97.5)),
        "cohen_kappa": 0.0,
        "auc_roc": None,
        "cost_usd": 0.0,
        "null_rate": 0.0,
        "n_runs": n_runs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DIEBOLD-MARIANO TEST
# ═══════════════════════════════════════════════════════════════════════════════

def diebold_mariano_test(y_true_labels: np.ndarray,
                          pred_a: np.ndarray,
                          pred_b: np.ndarray,
                          returns: np.ndarray = None) -> dict:
    """
    Diebold-Mariano (DM) test for equal predictive ability.
    H0: E[L(pred_a) - L(pred_b)] = 0  (both equally good).
    Loss: 0-1 loss (default) or squared-error on return magnitude.
    Ref: Diebold & Mariano (1995), JBES 13(3), 253-263.
    """
    from scipy.stats import ttest_1samp
    loss_type = getattr(config, "DM_LOSS", "01")
    mask = (pd.notna(pred_a) & pd.notna(pred_b) & pd.notna(y_true_labels))
    if returns is not None:
        mask = mask & pd.notna(returns)
    mask = np.array(mask)
    yt = np.array(y_true_labels)[mask]
    pa = np.array(pred_a)[mask]
    pb = np.array(pred_b)[mask]
    if len(yt) < 10:
        return {"error": f"Insufficient samples ({len(yt)} < 10)"}
    if loss_type == "se" and returns is not None:
        ret = np.array(returns)[mask]
        loss_a = (pa != yt).astype(float) * ret ** 2
        loss_b = (pb != yt).astype(float) * ret ** 2
    else:
        loss_a = (pa != yt).astype(float)
        loss_b = (pb != yt).astype(float)
        loss_type = "01"
    d = loss_a - loss_b
    t_stat, p_value = ttest_1samp(d, popmean=0.0)
    interp = (
        "pred_b significantly outperforms pred_a" if (p_value < 0.05 and t_stat > 0) else
        "pred_a significantly outperforms pred_b" if (p_value < 0.05 and t_stat < 0) else
        "no significant difference in predictive ability"
    )
    return {
        "dm_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "p_value_one_sided": round(float(p_value / 2), 6),
        "loss_diff_mean": round(float(d.mean()), 6),
        "loss_diff_std": round(float(d.std()), 6),
        "n": int(len(d)),
        "loss_type": loss_type,
        "significant": bool(p_value < config.SIGNIFICANCE_ALPHA),
        "interpretation": interp,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED CALIBRATION ERROR + RELIABILITY DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(confidences: np.ndarray,
                                correctness: np.ndarray,
                                n_bins: int = None) -> dict:
    """ECE = sum_b (|B_b|/N)*|acc(B_b)-conf(B_b)|. Returns per-bin data."""
    n_bins = n_bins or getattr(config, "ECE_N_BINS", 10)
    n = len(confidences)
    if n == 0:
        return {"ece": float("nan"), "mce": float("nan"), "bins": []}
    confidences = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    correctness = np.array(correctness, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_out, ece, mce = [], 0.0, 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = ((confidences >= lo) & (confidences <= hi)
                if i == n_bins - 1 else (confidences >= lo) & (confidences < hi))
        n_bin = int(mask.sum())
        if n_bin == 0:
            bins_out.append({"bin_lower": round(lo, 3), "bin_upper": round(hi, 3),
                              "n": 0, "avg_confidence": None, "avg_accuracy": None, "gap": None})
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc  = float(correctness[mask].mean())
        gap      = abs(avg_acc - avg_conf)
        ece     += (n_bin / n) * gap
        mce      = max(mce, gap)
        bins_out.append({"bin_lower": round(lo, 3), "bin_upper": round(hi, 3),
                         "n": n_bin, "avg_confidence": round(avg_conf, 4),
                         "avg_accuracy": round(avg_acc, 4), "gap": round(gap, 4)})
    overall_conf = float(confidences.mean())
    overall_acc  = float(correctness.mean())
    diagnosis = ("overconfident" if overall_conf > overall_acc + 0.02 else
                 "underconfident" if overall_conf < overall_acc - 0.02 else
                 "well-calibrated")
    return {"ece": round(ece, 6), "mce": round(mce, 6),
            "overall_confidence": round(overall_conf, 4),
            "overall_accuracy": round(overall_acc, 4),
            "diagnosis": diagnosis, "bins": bins_out,
            "n_total": n, "n_bins": n_bins}


def plot_reliability_diagram(ece_result: dict, title: str, save_path: str):
    """Reliability diagram + confidence histogram side-by-side."""
    bins = [b for b in ece_result["bins"] if b["n"] > 0]
    if not bins:
        return
    avg_confs = [b["avg_confidence"] for b in bins]
    avg_accs  = [b["avg_accuracy"]   for b in bins]
    ns        = [b["n"]              for b in bins]
    total_n   = ece_result["n_total"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect calibration")
    ax.scatter(avg_confs, avg_accs,
               s=[max(20, 200 * n / total_n) for n in ns],
               c=avg_accs, cmap="RdYlGn", vmin=0, vmax=1,
               edgecolors="black", linewidths=0.5, zorder=3)
    ax.plot(avg_confs, avg_accs, "b-o", ms=4, alpha=0.6)
    ax.fill_between(avg_confs, avg_confs, avg_accs, alpha=0.15, color="red",
                    label=f"ECE = {ece_result['ece']:.4f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Mean Confidence"); ax.set_ylabel("Fraction Correct")
    ax.set_title(f"Reliability Diagram\n{title}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax2 = axes[1]
    lowers = [b["bin_lower"] for b in ece_result["bins"]]
    counts = [b["n"]         for b in ece_result["bins"]]
    colors = ["#e74c3c" if (b.get("gap") or 0) > 0.1 else "#3498db"
              for b in ece_result["bins"]]
    ax2.bar(lowers, counts, width=1.0 / ece_result["n_bins"],
            align="edge", color=colors, edgecolor="white", lw=0.5)
    ax2.set_xlabel("Confidence"); ax2.set_ylabel("Count")
    ax2.set_title(f"Confidence Distribution\nDiagnosis: {ece_result['diagnosis']}")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# COST-EFFECTIVENESS PARETO FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════

def plot_cost_pareto(all_metrics: dict, task: str, save_path: str):
    """
    Cost-effectiveness Pareto frontier: macro F1 vs estimated cost.
    Highlights experiments that are Pareto-optimal (undominated).
    """
    task_metrics = {k: v for k, v in all_metrics.items() if k.startswith(f"{task}/")}
    if not task_metrics:
        return
    names, costs, f1s, is_baseline = [], [], [], []
    for k, m in task_metrics.items():
        exp_name = k.split("/", 1)[1]
        cost = float(m.get("cost_usd") or 0.0)
        f1   = float(m.get("f1_macro") or 0.0)
        names.append(exp_name); costs.append(max(cost, 0.005))
        f1s.append(f1); is_baseline.append(exp_name in ("Majority-class", "Random"))
    costs = np.array(costs); f1s = np.array(f1s)
    pareto_mask = np.array([
        not any(costs[j] <= costs[i] and f1s[j] >= f1s[i]
                and (costs[j] < costs[i] or f1s[j] > f1s[i])
                for j in range(len(names)) if j != i)
        for i in range(len(names))
    ])
    pareto_idx = np.where(pareto_mask)[0][np.argsort(costs[pareto_mask])]
    def color(n):
        if "XGB" in n:    return "#2980b9"
        if "FinBERT" in n: return "#8e44ad"
        if "Ensemble" in n or "ensemble" in n: return "#e67e22"
        if n in ("Majority-class", "Random"):  return "#95a5a6"
        return "#27ae60"
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (nm, c, f) in enumerate(zip(names, costs, f1s)):
        mk = "*" if pareto_mask[i] else ("x" if is_baseline[i] else "o")
        sz = 180 if pareto_mask[i] else (90 if is_baseline[i] else 55)
        ax.scatter(c, f, c=color(nm), marker=mk, s=sz, zorder=3,
                   edgecolors="black" if pareto_mask[i] else "none", linewidths=0.8)
        kw = dict(textcoords="offset points", xytext=(5, 4))
        if pareto_mask[i] or is_baseline[i]:
            ax.annotate(nm, (c, f), fontsize=7.5, fontweight="bold" if pareto_mask[i] else "normal",
                        color=color(nm), **kw)
        else:
            ax.annotate(nm, (c, f), fontsize=6.5, color="gray", alpha=0.8,
                        textcoords="offset points", xytext=(4, 3))
    if len(pareto_idx) > 1:
        ax.step(costs[pareto_idx], f1s[pareto_idx], where="post",
                color="red", lw=1.5, ls="--", alpha=0.7, label="Pareto frontier")
    for nm, f in zip(names, f1s):
        if nm == "Majority-class":
            ax.axhline(f, color="#95a5a6", ls=":", lw=1.2, label=f"Majority-class F1={f:.3f}")
        if nm == "Random":
            ax.axhline(f, color="#bdc3c7", ls=":", lw=1.0, label=f"Random F1={f:.3f}")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel("Estimated API Cost (USD, symlog scale)", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title(f"Cost-Effectiveness Pareto Frontier — {task.title()}\n"
                 "★ = Pareto-optimal", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(max(0, min(f1s) - 0.05), min(1.0, max(f1s) + 0.08))
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# POST-HOC CALIBRATION: TEMPERATURE SCALING + PLATT SCALING
# ═══════════════════════════════════════════════════════════════════════════════

def temperature_scale_confidences(confidences: np.ndarray,
                                   correctness: np.ndarray,
                                   n_bins: int = 10) -> tuple[float, np.ndarray]:
    """
    Temperature scaling for GPT-style agreement-rate confidences.

    Temperature scaling divides logit scores by a scalar T, learned by
    minimising NLL on the held-out predictions. Here we adapt it for
    binary agreement-rate confidences in [0,1] by learning T that
    minimises the gap between confidence and accuracy (equivalent to
    NLL minimisation under a Bernoulli model).

    Uses a simple grid search over T in [0.1, 5.0] since we have few
    calibration points (N=90 or 100).

    Returns:
        best_T    : learned temperature scalar
        cal_confs : calibrated confidences (conf / T, clipped to [0,1])
    """
    from scipy.special import expit  # sigmoid

    best_T   = 1.0
    best_nll = float("inf")

    for T in np.linspace(0.1, 5.0, 100):
        # Scale: treat confidence as a logit proxy; divide by T
        # clip to avoid log(0)
        cal = np.clip(confidences / T, 1e-7, 1 - 1e-7)
        nll = -np.mean(
            correctness * np.log(cal) + (1 - correctness) * np.log(1 - cal)
        )
        if nll < best_nll:
            best_nll = nll
            best_T   = T

    cal_confs = np.clip(confidences / best_T, 0.0, 1.0)
    return float(best_T), cal_confs


def platt_scale_confidences(confidences: np.ndarray,
                             correctness: np.ndarray) -> tuple[object, np.ndarray]:
    """
    Platt scaling for XGBoost max-probability confidences.

    Platt scaling fits a logistic regression on the raw model scores
    to map them to calibrated probabilities. This is the standard
    post-hoc calibration method for SVMs and tree ensembles.

    Requires sklearn LogisticRegression. Uses leave-one-out cross-validation
    to avoid overfitting on the small calibration set.

    Returns:
        platt_model : fitted LogisticRegression (for reporting)
        cal_confs   : calibrated probabilities
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import LeaveOneOut

    if len(confidences) < 5:
        return None, confidences  # too few samples to fit

    X = confidences.reshape(-1, 1)
    y = correctness.astype(int)

    # LOO-CV Platt scaling (robust for small N)
    cal_confs = np.zeros_like(confidences, dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        try:
            lr = LogisticRegressionCV(cv=3, max_iter=500, random_state=42)
            lr.fit(X[train_idx], y[train_idx])
            cal_confs[test_idx] = lr.predict_proba(X[test_idx])[:, 1]
        except Exception:
            cal_confs[test_idx] = confidences[test_idx]

    # Fit final model on all data for reporting
    try:
        final_lr = LogisticRegressionCV(cv=3, max_iter=500, random_state=42)
        final_lr.fit(X, y)
    except Exception:
        final_lr = None

    return final_lr, cal_confs


def apply_and_report_calibration(ece_before: dict, cal_confs: np.ndarray,
                                  correctness: np.ndarray, label: str,
                                  method: str, save_path: str) -> dict:
    """
    Compute ECE after calibration, plot before/after reliability diagrams,
    and return a summary dict for the calibration report.
    """
    ece_after = expected_calibration_error(cal_confs, correctness)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ece, title_suffix in zip(
        axes,
        [ece_before, ece_after],
        ["Before calibration", f"After {method}"]
    ):
        bins = [b for b in ece["bins"] if b["n"] > 0]
        if not bins:
            continue
        avg_confs = [b["avg_confidence"] for b in bins]
        avg_accs  = [b["avg_accuracy"]   for b in bins]
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfect")
        ax.scatter(avg_confs, avg_accs, s=60, c=avg_accs, cmap="RdYlGn",
                   vmin=0, vmax=1, edgecolors="black", linewidths=0.5, zorder=3)
        ax.plot(avg_confs, avg_accs, "b-o", ms=4, alpha=0.6)
        ax.fill_between(avg_confs, avg_confs, avg_accs, alpha=0.15, color="red",
                        label=f"ECE={ece['ece']:.4f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Mean Confidence"); ax.set_ylabel("Fraction Correct")
        ax.set_title(f"{label}\n{title_suffix}"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return {
        "ece_before": ece_before["ece"],
        "ece_after":  ece_after["ece"],
        "ece_reduction": round(ece_before["ece"] - ece_after["ece"], 6),
        "ece_reduction_pct": round(
            100 * (ece_before["ece"] - ece_after["ece"]) / max(ece_before["ece"], 1e-9), 2
        ),
        "method": method,
        "diagnosis_before": ece_before["diagnosis"],
        "diagnosis_after":  ece_after["diagnosis"],
        "n_total": ece_after["n_total"],
    }


def main():
    print("=" * 70)
    print("STEP 8: EVALUATION & PAPER FIGURES")
    print("=" * 70)

    # -- Load labels ------------------------------------------------------
    print("\n[1/6] Loading labels and predictions...")
    labels = pd.read_csv(config.LABELS_DIR / "labels_100.csv")
    labels = labels.set_index("symbol")

    # -- Load all predictions ---------------------------------------------
    # GPT base + hybrid
    all_preds = pd.read_csv(config.RESULTS_DIR / "all_predictions.csv")
    # Deduplicate: keep last row per experiment/task/symbol (handles re-runs)
    all_preds = all_preds.drop_duplicates(
        subset=["experiment", "task", "symbol"], keep="last"
    ).reset_index(drop=True)

    # Prompt ensemble predictions (from 06b_prompt_ensemble.py)
    ensemble_prompt_path = config.RESULTS_DIR / "ensemble_prompt_predictions.csv"
    if ensemble_prompt_path.exists():
        ensemble_prompt_df = pd.read_csv(ensemble_prompt_path)[
            ["experiment", "task", "symbol", "prediction"]
        ]
        # Add dummy token columns so concat works cleanly
        ensemble_prompt_df["input_tokens"] = 0
        ensemble_prompt_df["output_tokens"] = 0
        ensemble_prompt_df["raw_response"] = ""
        all_preds = pd.concat([all_preds, ensemble_prompt_df], ignore_index=True)
        print(f"  Loaded prompt ensemble predictions: "
              f"{ensemble_prompt_df['experiment'].unique().tolist()}")
    else:
        print("  NOTE: ensemble_prompt_predictions.csv not found. "
              "Run 06b_prompt_ensemble.py to add prompt ensemble results.")

    # XGBoost
    xgb_results_path = config.RESULTS_DIR / "xgb_results.json"
    xgb_results = {}
    if xgb_results_path.exists():
        with open(xgb_results_path) as f:
            xgb_results = json.load(f)

    # XGB predictions
    xgb_pred_dfs = {}
    for task in config.TASKS:
        path = config.RESULTS_DIR / f"xgb_predictions_{task}.csv"
        if path.exists():
            xgb_pred_dfs[task] = pd.read_csv(path).set_index("symbol")

    # XGB-fullcorpus predictions (from 05b_xgboost_full_corpus.py)
    xgb_fc_pred_dfs = {}
    xgb_fc_results = {}
    xgb_fc_json_path = config.RESULTS_DIR / "xgb_fullcorpus_results.json"
    if xgb_fc_json_path.exists():
        with open(xgb_fc_json_path) as f:
            xgb_fc_results = json.load(f)
        print(f"  Loaded XGB-fullcorpus results JSON")
    for task in config.TASKS:
        fc_path = config.RESULTS_DIR / f"xgb_fullcorpus_predictions_{task}.csv"
        if fc_path.exists():
            xgb_fc_pred_dfs[task] = pd.read_csv(fc_path).set_index("symbol")
            print(f"  Loaded XGB-fullcorpus predictions ({task}): "
                  f"{len(xgb_fc_pred_dfs[task])} samples")

    # FinBERT predictions (from 05c_finbert_baseline.py)
    finbert_pred_df = None
    finbert_results = {}
    finbert_path = config.RESULTS_DIR / "finbert_predictions.csv"
    finbert_json_path = config.RESULTS_DIR / "finbert_results.json"
    if finbert_path.exists():
        finbert_pred_df = pd.read_csv(finbert_path).set_index("symbol")
        print(f"  Loaded FinBERT predictions: {len(finbert_pred_df)} samples")
    if finbert_json_path.exists():
        with open(finbert_json_path) as f:
            finbert_results = json.load(f)
        print(f"  Loaded FinBERT results: F1={finbert_results.get('f1_macro', 'N/A'):.4f}")

    # Ensemble
    ensemble_path = config.RESULTS_DIR / "ensemble_predictions.csv"
    ensemble_df = pd.read_csv(ensemble_path) if ensemble_path.exists() else pd.DataFrame()

    # -- Compute metrics for all experiments ------------------------------
    print("\n[2/6] Computing metrics...")
    all_metrics = {}

    for task in config.TASKS:
        label_col = f"label_{task}"
        label_names = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]

        valid_labels = labels[labels[label_col].notna()]
        y_true = valid_labels[label_col]
        y_true_arr = np.array(y_true)

        # --- Baselines ---
        maj_m = compute_majority_baseline(y_true_arr, label_names)
        all_metrics[f"{task}/Majority-class"] = maj_m
        rnd_m = compute_random_baseline(y_true_arr, label_names)
        all_metrics[f"{task}/Random"] = rnd_m
        print(f"  {task} baselines: majority={maj_m['f1_macro']:.3f} "
              f"(class={maj_m['majority_class']}), "
              f"random={rnd_m['f1_macro']:.3f}+/-{rnd_m['f1_macro_std']:.3f}")

        # --- XGBoost metrics ---
        # Each feat_set reads from its own dedicated prediction column so
        # XGB-full and XGB-sentiment-only are evaluated independently.
        for feat_set in ["full", "sentiment_only"]:
            exp_name = "XGB-sentiment-only" if feat_set == "sentiment_only" else "XGB-full"
            # Column written by 05_xgboost_baseline.py:
            #   full          -> xgb_pred_{task}
            #   sentiment_only -> xgb_pred_sentiment_only_{task}
            pred_col = (
                f"xgb_pred_sentiment_only_{task}"
                if feat_set == "sentiment_only"
                else f"xgb_pred_{task}"
            )
            if task in xgb_pred_dfs:
                xgb_df = xgb_pred_dfs[task]
                if pred_col not in xgb_df.columns:
                    print(f"  WARNING: column '{pred_col}' not found in "
                          f"xgb_predictions_{task}.csv — skipping {exp_name}. "
                          f"Re-run 05_xgboost_baseline.py to generate it.")
                    continue
                common = y_true.index.intersection(xgb_df.index)
                if len(common) > 0:
                    metrics = compute_metrics(
                        y_true.loc[common],
                        xgb_df.loc[common, pred_col],
                        label_names
                    )
                    metrics["cost_usd"] = 0.0  # XGBoost is free
                    # Fold-level mean ± std from nested CV results JSON
                    xgb_key = f"{task}_XGB-{feat_set}"
                    if xgb_key in xgb_results and "fold_results" in xgb_results[xgb_key]:
                        fold_f1s = [fr.get("f1_macro", 0)
                                    for fr in xgb_results[xgb_key]["fold_results"]]
                        metrics["f1_macro_mean"] = float(np.mean(fold_f1s))
                        metrics["f1_macro_std"]  = float(np.std(fold_f1s))
                    all_metrics[f"{task}/{exp_name}"] = metrics

        # --- XGB-fullcorpus metrics (from 05b) ---
        if task in xgb_fc_pred_dfs:
            fc_df = xgb_fc_pred_dfs[task]
            # Determine prediction column name — try common patterns
            pred_col = None
            for candidate in [f"xgb_pred_{task}", "prediction", f"pred_{task}"]:
                if candidate in fc_df.columns:
                    pred_col = candidate
                    break
            if pred_col is None:
                # Fallback: use last column that looks like a prediction
                for col in fc_df.columns:
                    if "pred" in col.lower():
                        pred_col = col
                        break
            if pred_col is not None:
                common = y_true.index.intersection(fc_df.index)
                if len(common) > 0:
                    metrics = compute_metrics(
                        y_true.loc[common],
                        fc_df.loc[common, pred_col],
                        label_names
                    )
                    metrics["cost_usd"] = 0.0
                    all_metrics[f"{task}/XGB-fullcorpus"] = metrics
                    print(f"    XGB-fullcorpus/{task}: F1={metrics['f1_macro']:.4f} "
                          f"Acc={metrics['accuracy']:.4f} "
                          f"(n={len(common)})")

        # --- FinBERT metrics (binary only) ---
        if task == "binary" and finbert_pred_df is not None and finbert_results:
            common = y_true.index.intersection(finbert_pred_df.index)
            if len(common) > 0:
                metrics = compute_metrics(
                    y_true.loc[common],
                    finbert_pred_df.loc[common, "finbert_pred_binary"],
                    label_names
                )
                metrics["cost_usd"] = 0.0
                # Use true probability-based AUC from finbert_results JSON
                if "auc_roc" in finbert_results and finbert_results["auc_roc"]:
                    metrics["auc_roc"] = finbert_results["auc_roc"]
                    metrics["auc_source"] = "oof_probabilities"
                # Fold-level mean ± std from FinBERT results JSON
                if "fold_results" in finbert_results:
                    fold_f1s_fb = [fr.get("f1_macro", 0)
                                   for fr in finbert_results["fold_results"]]
                    metrics["f1_macro_mean"] = float(np.mean(fold_f1s_fb))
                    metrics["f1_macro_std"]  = float(np.std(fold_f1s_fb))
                all_metrics[f"{task}/FinBERT-finetuned"] = metrics

        # --- GPT & Hybrid metrics ---
        gpt_experiments = all_preds[all_preds["task"] == task]["experiment"].unique()
        for exp_name in gpt_experiments:
            exp_preds = all_preds[
                (all_preds["task"] == task) & (all_preds["experiment"] == exp_name)
            ].set_index("symbol")

            common = y_true.index.intersection(exp_preds.index)
            if len(common) > 0:
                metrics = compute_metrics(
                    y_true.loc[common],
                    exp_preds.loc[common, "prediction"],
                    label_names
                )
                # Estimate cost
                total_tokens = exp_preds.loc[common, "input_tokens"].sum()
                metrics["cost_usd"] = round(total_tokens / 1_000_000 * 2.50, 4)
                n_null = int(exp_preds["prediction"].isna().sum())
                metrics["null_rate"] = round(n_null / max(len(exp_preds), 1), 4)
                all_metrics[f"{task}/{exp_name}"] = metrics

        # --- Ensemble metrics ---
        # Ensemble-XGB rows now use confidence-weighted voting (not hard
        # XGB tiebreak), so they are no longer structurally biased toward
        # XGB.  Evaluate all ensemble variants; keep top 6 for reporting.
        if not ensemble_df.empty:
            ens_experiments = ensemble_df[ensemble_df["task"] == task]["experiment"].unique()
            for ens_name in ens_experiments[:6]:
                ens_preds = ensemble_df[
                    (ensemble_df["task"] == task) & (ensemble_df["experiment"] == ens_name)
                ].set_index("symbol")
                common = y_true.index.intersection(ens_preds.index)
                if len(common) > 0:
                    metrics = compute_metrics(
                        y_true.loc[common],
                        ens_preds.loc[common, "prediction"],
                        label_names
                    )
                    metrics["cost_usd"] = 0.0  # ensembles reuse existing predictions
                    all_metrics[f"{task}/{ens_name}"] = metrics

    # -- Print results table ----------------------------------------------
    print("\n[3/6] Results summary:")

    for task in config.TASKS:
        print(f"\n  {'=' * 60}")
        print(f"  TASK: {task.upper()}")
        print(f"  {'=' * 60}")
        print(f"  {'Experiment':<30} {'F1':>6} {'':>7} {'Acc':>6} {'AUC':>6} {'Kappa':>6} {'CI':>15} {'Cost':>8}")
        print(f"  {'':30} {'':>6} {'(±std)':>7} {'':>6} {'':>6} {'':>6} {'':>15} {'':>8}")
        print(f"  {'-' * 90}")

        task_metrics = {k: v for k, v in all_metrics.items() if k.startswith(f"{task}/")}
        # Sort by F1
        sorted_exps = sorted(task_metrics.items(), key=lambda x: x[1].get("f1_macro", 0), reverse=True)

        for exp_key, m in sorted_exps:
            exp_name = exp_key.split("/", 1)[1]
            f1 = m.get("f1_macro", 0)
            acc = m.get("accuracy", 0)
            kappa = m.get("cohen_kappa", 0)
            ci_lo = m.get("f1_macro_ci_lower", 0)
            ci_hi = m.get("f1_macro_ci_upper", 0)
            cost = m.get("cost_usd", 0)
            auc = m.get("auc_roc")
            auc_str = f"{auc:.3f}" if auc is not None else "  N/A"
            f1_std = m.get("f1_macro_std")
            std_str = f"±{f1_std:.3f}" if f1_std is not None else "      "
            print(f"  {exp_name:<30} {f1:>6.3f}{std_str:>7} {acc:>6.3f} {auc_str:>6} {kappa:>6.3f} "
                  f"[{ci_lo:.3f},{ci_hi:.3f}] ${cost:>7.2f}")

    # -- McNemar's test + Holm-Bonferroni correction ----------------------
    print("\n[4/6] Statistical significance (McNemar's test + Holm-Bonferroni)...")
    significance_results = {}

    for task in config.TASKS:
        label_col = f"label_{task}"
        label_names = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]
        valid_labels = labels[labels[label_col].notna()]
        y_true_series = valid_labels[label_col]

        # Collect all prediction vectors
        pred_vectors = {}

        # XGBoost
        if task in xgb_pred_dfs:
            xgb_df = xgb_pred_dfs[task]
            common = y_true_series.index.intersection(xgb_df.index)
            pred_vectors["XGB-full"] = xgb_df.loc[common, f"xgb_pred_{task}"]

        # XGB-fullcorpus
        if task in xgb_fc_pred_dfs:
            fc_df = xgb_fc_pred_dfs[task]
            pred_col = None
            for candidate in [f"xgb_pred_{task}", "prediction", f"pred_{task}"]:
                if candidate in fc_df.columns:
                    pred_col = candidate
                    break
            if pred_col is None:
                for col in fc_df.columns:
                    if "pred" in col.lower():
                        pred_col = col
                        break
            if pred_col is not None:
                common = y_true_series.index.intersection(fc_df.index)
                if len(common) > 0:
                    pred_vectors["XGB-fullcorpus"] = fc_df.loc[common, pred_col]

        # GPT and Ensemble experiments for significance testing
        gpt_experiments = all_preds[all_preds["task"] == task]["experiment"].unique()
        for exp_name in gpt_experiments:
            exp_preds = all_preds[
                (all_preds["task"] == task) & (all_preds["experiment"] == exp_name)
            ].set_index("symbol")
            common = y_true_series.index.intersection(exp_preds.index)
            if len(common) > 0:
                pred_vectors[exp_name] = exp_preds.loc[common, "prediction"]

        # Pairwise McNemar
        # Always include key comparisons in significance tests
        common_idx = y_true_series.index
        priority_pairs = []
        if "GPT-disagreement-ensemble" in pred_vectors:
            for other in ["GPT-zero", "XGB-full", "GPT-zero-CoT"]:
                if other in pred_vectors:
                    priority_pairs.append(
                        ("GPT-disagreement-ensemble", other)
                    )
        # XGB-fullcorpus vs key experiments
        if "XGB-fullcorpus" in pred_vectors:
            for other in ["XGB-full", "GPT-two-stage-CoT", "GPT-few",
                          "GPT-few-CoT", "GPT-zero", "GPT-contrastive",
                          "GPT-enhanced-ensemble",
                          "GPT-disagreement-ensemble"]:
                if other in pred_vectors:
                    pair = ("XGB-fullcorpus", other)
                    if pair not in priority_pairs:
                        priority_pairs.append(pair)
        all_pairs = priority_pairs + [
            (a, b) for a, b in list(combinations(pred_vectors.keys(), 2))
            if (a, b) not in priority_pairs
        ]
        for a, b in all_pairs[:35]:
            idx = common_idx.intersection(pred_vectors[a].index).intersection(pred_vectors[b].index)
            if len(idx) < 10:
                continue
            yt = np.array(y_true_series.loc[idx])
            pa = np.array(pred_vectors[a].loc[idx])
            pb = np.array(pred_vectors[b].loc[idx])

            # Remove NaN predictions
            mask = pd.notna(pa) & pd.notna(pb)
            if mask.sum() < 10:
                continue

            p_val = mcnemar_test(yt[mask], pa[mask], pb[mask])
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            significance_results[f"{task}/{a}_vs_{b}"] = p_val
            if sig:
                print(f"    {task}: {a} vs {b}: p={p_val:.4f} {sig}")

    # -- Holm-Bonferroni multiple testing correction ----------------------
    if significance_results:
        print("\n  Holm-Bonferroni correction:")
        sorted_tests = sorted(significance_results.items(), key=lambda x: x[1])
        m_tests = len(sorted_tests)
        corrected_results = {}
        any_rejected = False
        for rank_i, (pair, raw_p) in enumerate(sorted_tests, 1):
            adjusted_alpha = config.SIGNIFICANCE_ALPHA / (m_tests - rank_i + 1)
            rejected = raw_p <= adjusted_alpha and not any_rejected is False
            # Holm procedure: reject while p_i <= alpha / (m - i + 1), stop on first non-rejection
            if not any_rejected:
                rejected = raw_p <= adjusted_alpha
                if not rejected:
                    any_rejected = True
            else:
                rejected = False
            corrected_results[pair] = {
                "raw_p": round(raw_p, 6),
                "holm_threshold": round(adjusted_alpha, 6),
                "significant_corrected": rejected,
            }
            if rejected:
                print(f"    {pair}: p={raw_p:.4f} <= {adjusted_alpha:.4f} -> SIGNIFICANT")

        # Overwrite significance_results with corrected versions
        significance_results = corrected_results
        print(f"  {sum(1 for v in corrected_results.values() if v['significant_corrected'])}"
              f"/{m_tests} tests significant after Holm-Bonferroni correction")

    # -- [4c/6] Diebold-Mariano tests ----------------------------------------
    print("\n[4c/6] Diebold-Mariano tests (equal predictive ability)...")
    dm_results = {}

    for task in config.TASKS:
        label_col  = f"label_{task}"
        valid_lbl  = labels[labels[label_col].notna()]
        y_true_dm  = valid_lbl[label_col]
        returns_s  = labels.loc[y_true_dm.index, "return"] if "return" in labels.columns else None

        pred_vecs_dm = {}
        if task in xgb_pred_dfs:
            xdf = xgb_pred_dfs[task]
            common = y_true_dm.index.intersection(xdf.index)
            pred_vecs_dm["XGB-full"] = xdf.loc[common, f"xgb_pred_{task}"]
        for exp_name in all_preds[all_preds["task"] == task]["experiment"].unique():
            ep = all_preds[
                (all_preds["task"] == task) & (all_preds["experiment"] == exp_name)
            ].set_index("symbol")
            common = y_true_dm.index.intersection(ep.index)
            if len(common) > 0:
                pred_vecs_dm[exp_name] = ep.loc[common, "prediction"]

        # Every model vs majority-class baseline
        maj_cls = np.array(y_true_dm.value_counts().idxmax())
        maj_arr = np.full(len(y_true_dm), maj_cls)
        for exp_name, preds_dm in pred_vecs_dm.items():
            cidx = y_true_dm.index.intersection(preds_dm.index)
            if len(cidx) < 10:
                continue
            yt  = np.array(y_true_dm.loc[cidx])
            pa  = np.array(preds_dm.loc[cidx])
            ret = np.array(returns_s.loc[cidx]) if returns_s is not None else None
            res = diebold_mariano_test(yt, maj_arr[:len(cidx)], pa, returns=ret)
            key = f"{task}/{exp_name}_vs_Majority"
            dm_results[key] = res
            if isinstance(res, dict) and res.get("significant"):
                print(f"    {key}: DM={res['dm_statistic']:.3f} "
                      f"p={res['p_value']:.4f} -> {res['interpretation']}")

        # Priority pairwise comparisons
        for a, b in [("GPT-two-stage-CoT", "GPT-zero"),
                     ("GPT-few-CoT", "GPT-zero"),
                     ("GPT-xgb-inject", "XGB-full"),
                     ("GPT-disagreement-ensemble", "GPT-zero"),
                     ("GPT-disagreement-ensemble", "XGB-full")]:
            if a not in pred_vecs_dm or b not in pred_vecs_dm:
                continue
            cidx = (y_true_dm.index
                    .intersection(pred_vecs_dm[a].index)
                    .intersection(pred_vecs_dm[b].index))
            if len(cidx) < 10:
                continue
            yt  = np.array(y_true_dm.loc[cidx])
            pa  = np.array(pred_vecs_dm[a].loc[cidx])
            pb  = np.array(pred_vecs_dm[b].loc[cidx])
            ret = np.array(returns_s.loc[cidx]) if returns_s is not None else None
            res = diebold_mariano_test(yt, pa, pb, returns=ret)
            key = f"{task}/{a}_vs_{b}_DM"
            dm_results[key] = res
            if isinstance(res, dict) and res.get("significant"):
                print(f"    {key}: DM={res['dm_statistic']:.3f} "
                      f"p={res['p_value']:.4f} -> {res['interpretation']}")

    with open(config.RESULTS_DIR / "dm_test_results.json", "w") as f:
        json.dump(dm_results, f, indent=2, default=str)
    n_sig_dm = sum(1 for v in dm_results.values()
                   if isinstance(v, dict) and v.get("significant"))
    print(f"  DM tests: {len(dm_results)} pairs, {n_sig_dm} significant "
          f"(alpha={config.SIGNIFICANCE_ALPHA})")
    print(f"  Saved: {config.RESULTS_DIR / 'dm_test_results.json'}")

    # -- Generate figures -------------------------------------------------

    # -- [4b/6] Disagreement analysis — GPT-zero vs GPT-zero-CoT ---------
    print("\n[4b/6] Disagreement analysis (GPT-zero vs GPT-zero-CoT)...")

    disagreement_results = {}

    for task in config.TASKS:
        label_col = f"label_{task}"
        y_true_series = labels[label_col].dropna()

        # Pull predictions for GPT-zero and GPT-zero-CoT
        def get_preds(exp_name):
            rows = all_preds[
                (all_preds["task"] == task) &
                (all_preds["experiment"] == exp_name)
            ].set_index("symbol")["prediction"]
            return rows

        zero_preds = get_preds("GPT-zero")
        cot_preds  = get_preds("GPT-zero-CoT")

        # Common symbols with valid predictions from both and a true label
        common = (y_true_series.index
                  .intersection(zero_preds.index)
                  .intersection(cot_preds.index))

        if len(common) == 0:
            print(f"  {task}: no common samples found, skipping.")
            continue

        yt     = y_true_series.loc[common]
        y_zero = zero_preds.loc[common]
        y_cot  = cot_preds.loc[common]

        # Agreement / disagreement masks
        agree_mask    = (y_zero == y_cot)
        disagree_mask = ~agree_mask

        n_agree    = int(agree_mask.sum())
        n_disagree = int(disagree_mask.sum())
        print(f"  {task}: {n_agree} agree / {n_disagree} disagree "
              f"({100*n_disagree/len(common):.1f}% disagreement rate)")

        # Metrics on agreement subset (drop NaN predictions)
        def safe_f1(yt_sub, yp_sub):
            mask = pd.notna(yt_sub) & pd.notna(yp_sub)
            if mask.sum() < 2:
                return float("nan")
            return f1_score(
                np.array(yt_sub[mask]), np.array(yp_sub[mask]),
                average="macro", zero_division=0,
            )

        # On agreement samples: both models agree — use that agreed label
        if n_agree > 0:
            f1_agree_zero = safe_f1(yt[agree_mask], y_zero[agree_mask])
            f1_agree_cot  = safe_f1(yt[agree_mask], y_cot[agree_mask])
            # They're equal since preds are the same when they agree
            print(f"    Agreement subset   (N={n_agree}): "
                  f"F1={f1_agree_zero:.3f}")
        else:
            f1_agree_zero = float("nan")

        # On disagreement samples: compare each model's accuracy
        if n_disagree > 0:
            f1_dis_zero = safe_f1(yt[disagree_mask], y_zero[disagree_mask])
            f1_dis_cot  = safe_f1(yt[disagree_mask], y_cot[disagree_mask])
            print(f"    Disagreement subset (N={n_disagree}): "
                  f"GPT-zero F1={f1_dis_zero:.3f} | "
                  f"GPT-zero-CoT F1={f1_dis_cot:.3f}")
            better = "GPT-zero" if f1_dis_zero >= f1_dis_cot else "GPT-zero-CoT"
            print(f"    -> {better} is more accurate on hard samples")
        else:
            f1_dis_zero = f1_dis_cot = float("nan")

        # Per-class accuracy on disagreement subset (shows where they differ)
        if n_disagree > 0:
            label_names = sorted(yt.dropna().unique())
            print(f"    Disagreement breakdown by true class:")
            for cls in label_names:
                cls_mask = disagree_mask & (yt == cls)
                if cls_mask.sum() == 0:
                    continue
                zero_correct = int((y_zero[cls_mask] == yt[cls_mask]).sum())
                cot_correct  = int((y_cot[cls_mask] == yt[cls_mask]).sum())
                n_cls = int(cls_mask.sum())
                print(f"      {cls}: N={n_cls} | "
                      f"GPT-zero correct={zero_correct} | "
                      f"GPT-zero-CoT correct={cot_correct}")

        disagreement_results[task] = {
            "n_total":     len(common),
            "n_agree":     n_agree,
            "n_disagree":  n_disagree,
            "disagree_rate": round(n_disagree / len(common), 4),
            "f1_agree_subset":      round(f1_agree_zero, 4) if not np.isnan(f1_agree_zero) else None,
            "f1_disagree_gpt_zero": round(f1_dis_zero, 4) if not np.isnan(f1_dis_zero) else None,
            "f1_disagree_gpt_cot":  round(f1_dis_cot, 4) if not np.isnan(f1_dis_cot) else None,
            "disagree_symbols": common[disagree_mask].tolist(),
        }

        # -- Disagreement figure ------------------------------------------
        if n_disagree > 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Left: pie chart of agree vs disagree
            axes[0].pie(
                [n_agree, n_disagree],
                labels=[f"Agree (N={n_agree})", f"Disagree (N={n_disagree})"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.0f%%",
                startangle=90,
            )
            axes[0].set_title(f"GPT-zero vs GPT-zero-CoT\n{task.title()} Agreement")

            # Right: grouped bar — F1 on agree vs disagree subsets
            metrics_labels = ["Agreement subset", "Disagreement subset"]
            zero_scores = [
                f1_agree_zero if not np.isnan(f1_agree_zero) else 0,
                f1_dis_zero   if not np.isnan(f1_dis_zero) else 0,
            ]
            cot_scores = [
                f1_agree_cot  if "f1_agree_cot" in dir() and not np.isnan(f1_agree_cot) else zero_scores[0],
                f1_dis_cot    if not np.isnan(f1_dis_cot) else 0,
            ]
            x = np.arange(len(metrics_labels))
            w = 0.35
            axes[1].bar(x - w/2, zero_scores, w, label="GPT-zero", color="#3498db")
            axes[1].bar(x + w/2, cot_scores,  w, label="GPT-zero-CoT", color="#e67e22")
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(metrics_labels)
            axes[1].set_ylabel("Macro F1")
            axes[1].set_ylim(0, 1)
            axes[1].set_title(f"F1 by Agreement Status\n{task.title()}")
            axes[1].legend()

            plt.tight_layout()
            fig_path = config.FIGURES_DIR / f"disagreement_analysis_{task}.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"    Saved: {fig_path}")

    # Save disagreement results
    disag_path = config.RESULTS_DIR / "disagreement_analysis.json"
    with open(disag_path, "w") as f:
        json.dump(disagreement_results, f, indent=2, default=str)
    print(f"  Saved: {disag_path}")

    print("\n[5/6] Generating figures...")

    # -- Confusion matrices for key experiments --
    for task in config.TASKS:
        label_names = ["UP", "DOWN"] if task == "binary" else ["UP", "DOWN", "FLAT"]
        task_metrics = {k: v for k, v in all_metrics.items() if k.startswith(f"{task}/")}

        # Top 4 experiments by F1
        sorted_exps = sorted(task_metrics.items(),
                             key=lambda x: x[1].get("f1_macro", 0), reverse=True)[:4]

        fig, axes = plt.subplots(1, min(4, len(sorted_exps)),
                                 figsize=(5 * min(4, len(sorted_exps)), 4))
        if len(sorted_exps) == 1:
            axes = [axes]

        for i, (exp_key, m) in enumerate(sorted_exps):
            if i >= len(axes):
                break
            cm = np.array(m.get("confusion_matrix", []))
            if cm.size == 0:
                continue
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=label_names, yticklabels=label_names,
                        ax=axes[i])
            exp_name = exp_key.split("/", 1)[1]
            axes[i].set_title(f"{exp_name}\nF1={m['f1_macro']:.3f}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"confusion_matrices_{task}.png", dpi=150)
        plt.close()

    # -- Cost-effectiveness Pareto frontier --
    print("  Generating cost-effectiveness Pareto frontier plots...")
    for task in config.TASKS:
        plot_cost_pareto(
            all_metrics, task,
            str(config.FIGURES_DIR / f"cost_pareto_{task}.png")
        )
    print("  Saved cost_pareto_binary.png / cost_pareto_ternary.png")

    # -- [5b/6] Calibration: ECE + reliability diagrams --
    print("  Generating reliability diagrams (ECE)...")
    calibration_results = {}

    for task in config.TASKS:
        label_col = f"label_{task}"
        valid_lbl_cal = labels[labels[label_col].notna()]
        y_true_cal    = np.array(valid_lbl_cal[label_col])

        # GPT-zero-calibrated: parse agreement rate from raw_response as confidence
        cal_rows = all_preds[
            (all_preds["task"] == task) &
            (all_preds["experiment"] == "GPT-zero-calibrated")
        ].set_index("symbol")
        common_cal = valid_lbl_cal.index.intersection(cal_rows.index)

        if len(common_cal) >= 10:
            preds_cal   = np.array(cal_rows.loc[common_cal, "prediction"])
            true_cal    = np.array(valid_lbl_cal.loc[common_cal, label_col])
            correct_cal = (preds_cal == true_cal).astype(float)

            if "confidence" in cal_rows.columns:
                conf_arr = cal_rows.loc[common_cal, "confidence"].fillna(
                    1.0 / (2 if task == "binary" else 3)).values.astype(float)
            elif "raw_response" in cal_rows.columns:
                import re as _re
                def _parse_agr(raw):
                    m = _re.search(r"agreement=([0-9.]+)", str(raw))
                    return float(m.group(1)) if m else 1.0 / (2 if task == "binary" else 3)
                conf_arr = np.array([_parse_agr(r)
                                     for r in cal_rows.loc[common_cal, "raw_response"]])
            else:
                conf_arr = np.full(len(common_cal), 1.0 / (2 if task == "binary" else 3))

            ece_res = expected_calibration_error(conf_arr, correct_cal)
            calibration_results[f"{task}/GPT-zero-calibrated"] = ece_res
            plot_reliability_diagram(
                ece_res, f"GPT-zero-calibrated — {task.title()}",
                str(config.FIGURES_DIR / f"reliability_gpt_calibrated_{task}.png"))
            print(f"    {task}/GPT-zero-calibrated: ECE={ece_res['ece']:.4f} ({ece_res['diagnosis']})")

        # XGB-full: use max predicted probability as confidence
        if task in xgb_pred_dfs:
            xdf = xgb_pred_dfs[task]
            cidx = valid_lbl_cal.index.intersection(xdf.index)
            if (len(cidx) >= 10 and f"xgb_conf_{task}" in xdf.columns
                    and f"xgb_pred_{task}" in xdf.columns):
                conf_xgb = np.array(xdf.loc[cidx, f"xgb_conf_{task}"])
                pred_xgb = np.array(xdf.loc[cidx, f"xgb_pred_{task}"])
                true_xgb = np.array(valid_lbl_cal.loc[cidx, label_col])
                corr_xgb = (pred_xgb == true_xgb).astype(float)
                ece_xgb  = expected_calibration_error(conf_xgb, corr_xgb)
                calibration_results[f"{task}/XGB-full"] = ece_xgb
                plot_reliability_diagram(
                    ece_xgb, f"XGB-full — {task.title()}",
                    str(config.FIGURES_DIR / f"reliability_xgb_{task}.png"))
                print(f"    {task}/XGB-full: ECE={ece_xgb['ece']:.4f} ({ece_xgb['diagnosis']})")

    # ── Post-hoc calibration: temperature scaling (GPT) + Platt (XGB) ────
    print("  Applying post-hoc calibration (temperature scaling / Platt)...")
    posthoc_calibration = {}

    for task in config.TASKS:
        # Temperature scaling on GPT-zero-calibrated
        cal_key = f"{task}/GPT-zero-calibrated"
        if cal_key in calibration_results:
            cr = calibration_results[cal_key]
            bins = [b for b in cr["bins"] if b["n"] > 0]
            if bins:
                # Reconstruct conf and correctness arrays from bin data
                confs_gpt, correct_gpt = [], []
                for b in cr["bins"]:
                    if b["n"] > 0 and b["avg_confidence"] is not None:
                        confs_gpt.extend([b["avg_confidence"]] * b["n"])
                        correct_gpt.extend([b["avg_accuracy"]] * b["n"])
                confs_gpt   = np.array(confs_gpt)
                correct_gpt = np.array(correct_gpt)
                if len(confs_gpt) >= 5:
                    best_T, cal_confs_gpt = temperature_scale_confidences(
                        confs_gpt, correct_gpt)
                    result_gpt = apply_and_report_calibration(
                        ece_before=calibration_results[cal_key],
                        cal_confs=cal_confs_gpt,
                        correctness=correct_gpt,
                        label=f"GPT-zero-calibrated ({task})",
                        method=f"Temperature scaling (T={best_T:.2f})",
                        save_path=str(config.FIGURES_DIR /
                                      f"calibration_gpt_{task}.png")
                    )
                    result_gpt["temperature"] = best_T
                    posthoc_calibration[cal_key] = result_gpt
                    print(f"    {cal_key}: ECE {result_gpt['ece_before']:.4f} -> "
                          f"{result_gpt['ece_after']:.4f} "
                          f"({result_gpt['ece_reduction_pct']:+.1f}%) "
                          f"T={best_T:.2f}")

        # Platt scaling on XGB-full
        xgb_key = f"{task}/XGB-full"
        if xgb_key in calibration_results:
            cr_xgb = calibration_results[xgb_key]
            confs_xgb, correct_xgb = [], []
            for b in cr_xgb["bins"]:
                if b["n"] > 0 and b["avg_confidence"] is not None:
                    confs_xgb.extend([b["avg_confidence"]] * b["n"])
                    correct_xgb.extend([b["avg_accuracy"]] * b["n"])
            confs_xgb   = np.array(confs_xgb)
            correct_xgb = np.array(correct_xgb)
            if len(confs_xgb) >= 5:
                platt_model, cal_confs_xgb = platt_scale_confidences(
                    confs_xgb, correct_xgb)
                result_xgb = apply_and_report_calibration(
                    ece_before=calibration_results[xgb_key],
                    cal_confs=cal_confs_xgb,
                    correctness=correct_xgb,
                    label=f"XGB-full ({task})",
                    method="Platt scaling (LOO-CV LogisticRegression)",
                    save_path=str(config.FIGURES_DIR /
                                  f"calibration_xgb_{task}.png")
                )
                if platt_model is not None:
                    result_xgb["platt_coef"]      = float(platt_model.coef_[0][0])
                    result_xgb["platt_intercept"]  = float(platt_model.intercept_[0])
                posthoc_calibration[xgb_key] = result_xgb
                print(f"    {xgb_key}: ECE {result_xgb['ece_before']:.4f} -> "
                      f"{result_xgb['ece_after']:.4f} "
                      f"({result_xgb['ece_reduction_pct']:+.1f}%)")

    calibration_results["posthoc"] = posthoc_calibration

    with open(config.RESULTS_DIR / "calibration_results.json", "w") as f:
        json.dump(calibration_results, f, indent=2, default=str)
    print(f"  Saved calibration_results.json")

    # -- Label distribution --
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, task in enumerate(config.TASKS):
        label_col = f"label_{task}"
        vc = labels[label_col].value_counts()
        vc.plot(kind="bar", ax=axes[i], color=["#2ecc71", "#e74c3c", "#95a5a6"][:len(vc)])
        axes[i].set_title(f"Label Distribution ({task.title()})")
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "label_distribution.png", dpi=150)
    plt.close()

    # -- Ablation bar chart --
    for task in config.TASKS:
        task_metrics = {k: v for k, v in all_metrics.items() if k.startswith(f"{task}/")}
        sorted_exps = sorted(task_metrics.items(),
                             key=lambda x: x[1].get("f1_macro", 0), reverse=True)

        names = [k.split("/")[1] for k, _ in sorted_exps]
        f1s = [m.get("f1_macro", 0) for _, m in sorted_exps]
        ci_lo = [m.get("f1_macro_ci_lower", 0) for _, m in sorted_exps]
        ci_hi = [m.get("f1_macro_ci_upper", 0) for _, m in sorted_exps]
        errors = [[f - lo for f, lo in zip(f1s, ci_lo)],
                  [hi - f for f, hi in zip(f1s, ci_hi)]]

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(names)), f1s, xerr=errors, capsize=3,
                       color=["#3498db" if "XGB" in n else "#e67e22" if "Ensemble" in n
                              else "#2ecc71" for n in names])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Macro F1 Score")
        ax.set_title(f"Ablation Results — {task.title()} Classification")
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"ablation_{task}.png", dpi=150)
        plt.close()

    # -- Save complete results --------------------------------------------
    print("\n[6/6] Saving results...")

    # Save metrics JSON
    with open(config.RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save significance results
    with open(config.RESULTS_DIR / "significance_tests.json", "w") as f:
        json.dump(significance_results, f, indent=2, default=str)

    # Save DM test results (already saved inline, but ensure key exists)
    dm_path = config.RESULTS_DIR / "dm_test_results.json"
    if not dm_path.exists():
        with open(dm_path, "w") as f:
            json.dump({}, f)

    # Save calibration results (already saved inline, but ensure key exists)
    cal_path = config.RESULTS_DIR / "calibration_results.json"
    if not cal_path.exists():
        with open(cal_path, "w") as f:
            json.dump({}, f)

    # -- LaTeX ablation table (IEEE-quality with bold best, significance markers)
    for task in config.TASKS:
        task_metrics = {k: v for k, v in all_metrics.items() if k.startswith(f"{task}/")}
        sorted_exps = sorted(task_metrics.items(),
                             key=lambda x: x[1].get("f1_macro", 0), reverse=True)

        # Find best values for bolding
        best_f1 = max(m.get("f1_macro", 0) for _, m in sorted_exps) if sorted_exps else 0
        best_acc = max(m.get("accuracy", 0) for _, m in sorted_exps) if sorted_exps else 0

        # Categorize experiments for section headers
        categories = {
            "Classical ML": ["XGB-full", "XGB-sentiment-only", "XGB-fullcorpus", "XGB-PCA"],
            "Deep Learning": ["FinBERT-finetuned"],
            "Zero-shot LLM": ["GPT-zero", "GPT-zero-CoT", "GPT-zero-calibrated"],
            "Few-shot LLM": ["GPT-few", "GPT-few-CoT", "GPT-rag-few",
                             "GPT-few-sector", "GPT-few-CoT-sector"],
            "Two-stage LLM": ["GPT-two-stage", "GPT-two-stage-CoT"],
            "Hybrid (Ours)": ["GPT-feat-inject", "GPT-feat-only", "GPT-speaker-seg",
                              "GPT-contrastive", "GPT-xgb-inject"],
            "Ensembles": ["GPT-core-ensemble", "GPT-hybrid-ensemble",
                          "GPT-enhanced-ensemble", "GPT-full-ensemble",
                          "GPT-disagreement-ensemble"],
        }

        latex_lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\small",
            f"\\caption{{Comprehensive ablation results — {task} classification. "
            r"Best values in \textbf{bold}. $\dagger$ denotes significance "
            r"vs.\ GPT-zero (McNemar, $p < 0.05$, Holm-Bonferroni corrected).}",
            f"\\label{{tab:ablation_{task}}}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Experiment & Accuracy & Macro F1 & AUC-ROC & F1 95\% CI (BCa) & $\kappa$ & Null\% & Cost (\$) \\",
            r"\midrule",
        ]

        def fmt_bold(val, best, fmt=".3f"):
            s = f"{val:{fmt}}"
            return rf"\textbf{{{s}}}" if abs(val - best) < 1e-6 else s

        # Check significance vs GPT-zero
        def is_significant(exp_name):
            pair = f"GPT-zero_vs_{exp_name}"
            pair_rev = f"{exp_name}_vs_GPT-zero"
            for p in [pair, pair_rev]:
                if p in significance_results:
                    info = significance_results[p]
                    if isinstance(info, dict):
                        return info.get("significant_corrected", False)
                    return info < config.SIGNIFICANCE_ALPHA
            return False

        for cat_name, cat_exps in categories.items():
            cat_has_results = False
            cat_lines = []
            for exp_key, m in sorted_exps:
                exp_name = exp_key.split("/", 1)[1]
                if exp_name not in cat_exps:
                    continue
                cat_has_results = True

                acc = m.get("accuracy", 0)
                f1 = m.get("f1_macro", 0)
                ci_lo = m.get("f1_macro_ci_lower", 0)
                ci_hi = m.get("f1_macro_ci_upper", 0)
                kappa = m.get("cohen_kappa", 0)
                cost = m.get("cost_usd", 0)
                auc_val = m.get("auc_roc")
                auc_str = f"{auc_val:.3f}" if auc_val is not None else "--"

                sig_marker = r"$\dagger$" if is_significant(exp_name) else ""
                display_name = exp_name.replace("_", r"\_") + sig_marker

                null_rate_pct = m.get("null_rate", 0.0) * 100
                cat_lines.append(
                    f"  {display_name} & {fmt_bold(acc, best_acc)} & "
                    f"{fmt_bold(f1, best_f1)} & {auc_str} & "
                    f"[{ci_lo:.3f}, {ci_hi:.3f}] & {kappa:.3f} & "
                    f"{null_rate_pct:.1f}\\% & {cost:.2f} \\\\"
                )

            if cat_has_results:
                latex_lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{cat_name}}}}} \\")
                latex_lines.extend(cat_lines)

        latex_lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])

        latex_path = config.FIGURES_DIR / f"table_ablation_{task}.tex"
        with open(latex_path, "w", newline="\n", encoding="utf-8") as f:
            f.write("\n".join(latex_lines))
        print(f"  LaTeX table: {latex_path}")

    # -- Sensitivity analysis figure (threshold sweep) --------------------
    print("  Generating sensitivity analysis figure...")
    sensitivity_path = config.LABELS_DIR / "sensitivity_analysis.csv"
    if sensitivity_path.exists():
        sens_df = pd.read_csv(sensitivity_path)
        if not sens_df.empty and "threshold" in sens_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for i, metric in enumerate(["n_up", "n_down"]):
                if metric in sens_df.columns:
                    axes[i].plot(sens_df["threshold"], sens_df[metric], "o-",
                                 linewidth=2, markersize=6)
                    axes[i].set_xlabel("Threshold (%)")
                    axes[i].set_ylabel("Count")
                    axes[i].set_title(f"{metric.replace('_', ' ').title()} vs Threshold")
                    axes[i].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(config.FIGURES_DIR / "sensitivity_threshold.png", dpi=200)
            plt.close()
            print(f"  Saved sensitivity_threshold.png")

    # -- Radar/spider chart for top experiments ----------------------------
    print("  Generating radar chart...")
    for task in config.TASKS:
        task_metrics = {k: v for k, v in all_metrics.items() if k.startswith(f"{task}/")}
        top_exps = sorted(task_metrics.items(),
                          key=lambda x: x[1].get("f1_macro", 0), reverse=True)[:5]

        if len(top_exps) < 2:
            continue

        metric_names = ["Accuracy", "Macro F1", "Cohen's κ"]
        n_metrics = len(metric_names)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        colors = plt.cm.Set2(np.linspace(0, 1, len(top_exps)))

        for idx, (exp_key, m) in enumerate(top_exps):
            values = [
                m.get("accuracy", 0),
                m.get("f1_macro", 0),
                max(m.get("cohen_kappa", 0), 0),
            ]
            values += values[:1]
            exp_name = exp_key.split("/", 1)[1]
            ax.plot(angles, values, "o-", linewidth=2, label=exp_name,
                    color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f"Top Experiments — {task.title()}", fontsize=13, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"radar_top_experiments_{task}.png", dpi=200)
        plt.close()

    print(f"  Saved radar charts")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"  Metrics: {config.RESULTS_DIR / 'evaluation_metrics.json'}")
    print(f"  Significance: {config.RESULTS_DIR / 'significance_tests.json'}")
    print(f"  Figures: {config.FIGURES_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()