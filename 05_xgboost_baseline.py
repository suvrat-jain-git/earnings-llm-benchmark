"""
Step 5: XGBoost Baseline
Train XGBoost with nested cross-validation, Optuna hyperparameter tuning,
and SHAP analysis. Both binary and ternary classification tasks.
Ablation: full features vs L-M sentiment only.
"""
import pandas as pd
import numpy as np
import json
import joblib
import xgboost as xgb
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_feature_subsets(features_df: pd.DataFrame) -> dict:
    """Return feature column subsets for ablation experiments."""
    all_cols = [c for c in features_df.columns if c != "symbol"]

    # Sentiment-only: L-M features for all speaker segments
    lm_cols = [c for c in all_cols if "_lm_" in c]

    return {
        "full": all_cols,
        "sentiment_only": lm_cols,
        # PCA variant added at runtime (needs fit on training data per fold)
    }


def run_pca_nested_cv(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder,
                      task_name: str, n_classes: int,
                      n_components: int = None) -> dict:
    """
    Nested CV with PCA dimensionality reduction applied per fold
    (fit on train, transform test) to avoid data leakage.
    Addresses the 254-features-for-90-samples problem.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_components = n_components or getattr(config, "XGB_PCA_COMPONENTS", 20)
    n_components = min(n_components, X.shape[1], X.shape[0])

    print(f"    PCA: reducing {X.shape[1]} features -> {n_components} components")

    outer_cv = StratifiedKFold(
        n_splits=config.OUTER_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    inner_cv = StratifiedKFold(
        n_splits=config.INNER_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    all_preds = np.full(len(y), -1, dtype=int)
    all_probs = np.zeros((len(y), n_classes))
    fold_results = []

    objective_type = "multi:softprob" if n_classes > 2 else "binary:logistic"

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"    Outer fold {fold_idx + 1}/{config.OUTER_CV_FOLDS}...")
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # PCA fit on train only
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
        X_train = pca.fit_transform(X_train_scaled)
        X_test = pca.transform(X_test_scaled)

        explained_var = pca.explained_variance_ratio_.sum()
        print(f"      PCA explained variance: {explained_var:.3f}")

        # Inner CV with Optuna (on PCA features)
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            inner_scores = []
            for inner_train, inner_val in inner_cv.split(X_train, y_train):
                Xi_train, Xi_val = X_train[inner_train], X_train[inner_val]
                yi_train, yi_val = y_train[inner_train], y_train[inner_val]

                if n_classes > 2:
                    model = xgb.XGBClassifier(
                        objective=objective_type, num_class=n_classes,
                        eval_metric="mlogloss",
                        random_state=config.RANDOM_SEED, verbosity=0, **params,
                    )
                else:
                    n_neg = int(np.sum(yi_train == 0))
                    n_pos = int(np.sum(yi_train == 1))
                    spw = n_neg / max(n_pos, 1)
                    model = xgb.XGBClassifier(
                        objective=objective_type, eval_metric="logloss",
                        random_state=config.RANDOM_SEED,
                        verbosity=0, scale_pos_weight=spw, **params,
                    )

                model.fit(Xi_train, yi_train, verbose=False)
                preds = model.predict(Xi_val)
                inner_scores.append(f1_score(yi_val, preds, average="macro"))

            return np.mean(inner_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, show_progress_bar=False)
        best_params = study.best_params

        # Train outer fold model with best params
        if n_classes > 2:
            model = xgb.XGBClassifier(
                objective=objective_type, num_class=n_classes,
                eval_metric="mlogloss",
                random_state=config.RANDOM_SEED, verbosity=0, **best_params,
            )
        else:
            n_neg = int(np.sum(y_train == 0))
            n_pos = int(np.sum(y_train == 1))
            spw = n_neg / max(n_pos, 1)
            model = xgb.XGBClassifier(
                objective=objective_type, eval_metric="logloss",
                random_state=config.RANDOM_SEED,
                verbosity=0, scale_pos_weight=spw, **best_params,
            )

        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs

        fold_f1 = f1_score(y_test, preds, average="macro")
        fold_results.append({
            "fold": fold_idx, "f1_macro": fold_f1,
            "accuracy": accuracy_score(y_test, preds),
            "best_params": best_params,
            "pca_explained_variance": float(explained_var),
        })
        print(f"      Outer fold F1: {fold_f1:.4f}")

    valid_mask = all_preds >= 0
    overall_f1 = f1_score(y[valid_mask], all_preds[valid_mask], average="macro")
    overall_acc = accuracy_score(y[valid_mask], all_preds[valid_mask])

    label_names = label_encoder.classes_
    print(f"\n    Overall {task_name}/PCA-{n_components}:")
    print(f"      Macro F1:  {overall_f1:.4f}")
    print(f"      Accuracy:  {overall_acc:.4f}")
    print(classification_report(y[valid_mask], all_preds[valid_mask],
                                target_names=label_names))

    return {
        "task": task_name,
        "feature_set": f"PCA-{n_components}",
        "f1_macro": overall_f1,
        "accuracy": overall_acc,
        "fold_results": fold_results,
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "n_pca_components": n_components,
    }


def run_nested_cv(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder,
                  task_name: str, feature_set_name: str,
                  n_classes: int) -> dict:
    """
    Nested cross-validation: outer 5-fold eval, inner 3-fold Optuna tuning.
    """
    outer_cv = StratifiedKFold(
        n_splits=config.OUTER_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    inner_cv = StratifiedKFold(
        n_splits=config.INNER_CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )

    all_preds = np.full(len(y), -1, dtype=int)
    all_probs = np.zeros((len(y), n_classes))
    fold_results = []

    objective_type = "multi:softprob" if n_classes > 2 else "binary:logistic"

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"    Outer fold {fold_idx + 1}/{config.OUTER_CV_FOLDS}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV with Optuna
        def objective(trial):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            inner_scores = []
            for inner_train, inner_val in inner_cv.split(X_train, y_train):
                Xi_train, Xi_val = X_train[inner_train], X_train[inner_val]
                yi_train, yi_val = y_train[inner_train], y_train[inner_val]

                if n_classes > 2:
                    model = xgb.XGBClassifier(
                        objective=objective_type,
                        num_class=n_classes,
                        eval_metric="mlogloss",
                        
                        random_state=config.RANDOM_SEED,
                        verbosity=0,
                        **params,
                    )
                else:
                    # scale_pos_weight balances UP/DOWN class imbalance.
                    # Computed per inner-fold train split for correctness.
                    n_neg_inner = int(np.sum(yi_train == 0))
                    n_pos_inner = int(np.sum(yi_train == 1))
                    spw_inner = n_neg_inner / max(n_pos_inner, 1)
                    model = xgb.XGBClassifier(
                        objective=objective_type,
                        eval_metric="logloss",
                        
                        random_state=config.RANDOM_SEED,
                        verbosity=0,
                        scale_pos_weight=spw_inner,
                        **params,
                    )

                model.fit(Xi_train, yi_train, verbose=False)
                preds = model.predict(Xi_val)
                inner_scores.append(f1_score(yi_val, preds, average="macro"))

            return np.mean(inner_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, show_progress_bar=False)

        best_params = study.best_params
        print(f"      Best inner F1: {study.best_value:.4f}")

        # Train on full outer train with best params
        if n_classes > 2:
            model = xgb.XGBClassifier(
                objective=objective_type,
                num_class=n_classes,
                eval_metric="mlogloss",
                
                random_state=config.RANDOM_SEED,
                verbosity=0,
                **best_params,
            )
        else:
            # scale_pos_weight per outer fold
            n_neg_outer = int(np.sum(y_train == 0))
            n_pos_outer = int(np.sum(y_train == 1))
            spw_outer = n_neg_outer / max(n_pos_outer, 1)
            model = xgb.XGBClassifier(
                objective=objective_type,
                eval_metric="logloss",
                
                random_state=config.RANDOM_SEED,
                verbosity=0,
                scale_pos_weight=spw_outer,
                **best_params,
            )

        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)

        all_preds[test_idx] = preds
        all_probs[test_idx] = probs

        fold_f1 = f1_score(y_test, preds, average="macro")
        fold_results.append({
            "fold": fold_idx,
            "f1_macro": fold_f1,
            "accuracy": accuracy_score(y_test, preds),
            "best_params": best_params,
        })
        print(f"      Outer fold F1: {fold_f1:.4f}")

    # Overall metrics
    valid_mask = all_preds >= 0
    overall_f1 = f1_score(y[valid_mask], all_preds[valid_mask], average="macro")
    overall_acc = accuracy_score(y[valid_mask], all_preds[valid_mask])

    label_names = label_encoder.classes_
    report = classification_report(
        y[valid_mask], all_preds[valid_mask],
        target_names=label_names, output_dict=True
    )
    cm = confusion_matrix(y[valid_mask], all_preds[valid_mask])

    print(f"\n    Overall {task_name}/{feature_set_name}:")
    print(f"      Macro F1:  {overall_f1:.4f}")
    print(f"      Accuracy:  {overall_acc:.4f}")
    print(classification_report(y[valid_mask], all_preds[valid_mask],
                                target_names=label_names))

    return {
        "task": task_name,
        "feature_set": feature_set_name,
        "f1_macro": overall_f1,
        "accuracy": overall_acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "fold_results": fold_results,
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
    }


def train_final_model(X: np.ndarray, y: np.ndarray,
                      best_params: dict, n_classes: int) -> xgb.XGBClassifier:
    """Train a final model on all data with best params (for SHAP)."""
    objective_type = "multi:softprob" if n_classes > 2 else "binary:logistic"

    if n_classes > 2:
        model = xgb.XGBClassifier(
            objective=objective_type,
            num_class=n_classes,
            eval_metric="mlogloss",
            
            random_state=config.RANDOM_SEED,
            verbosity=0,
            **best_params,
        )
    else:
        # scale_pos_weight for final model (used for SHAP, not evaluation)
        n_neg_all = int(np.sum(y == 0))
        n_pos_all = int(np.sum(y == 1))
        spw_all = n_neg_all / max(n_pos_all, 1)
        model = xgb.XGBClassifier(
            objective=objective_type,
            eval_metric="logloss",
            
            random_state=config.RANDOM_SEED,
            verbosity=0,
            scale_pos_weight=spw_all,
            **best_params,
        )

    model.fit(X, y, verbose=False)
    return model


def main():
    print("=" * 70)
    print("STEP 5: XGBOOST BASELINE")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/4] Loading features and labels...")
    features_df = pd.read_parquet(config.FEATURES_DIR / "features_100.parquet")
    labels_df = pd.read_csv(config.LABELS_DIR / "labels_100.csv")

    # Merge on symbol
    labels_df = labels_df.set_index("symbol")
    merged = features_df.join(labels_df[["label_ternary", "label_binary"]], how="inner")

    print(f"  Merged samples: {len(merged)}")

    feature_subsets = get_feature_subsets(features_df)
    all_results = {}

    # ── Run experiments ──────────────────────────────────────────────────
    for task in config.TASKS:
        label_col = f"label_{task}"

        # Filter valid labels
        valid = merged[merged[label_col].notna()].copy()
        print(f"\n[2/4] Task: {task} — {len(valid)} valid samples")
        print(f"  Label distribution: {valid[label_col].value_counts().to_dict()}")

        le = LabelEncoder()
        y = le.fit_transform(valid[label_col].values)
        n_classes = len(le.classes_)
        print(f"  Classes: {list(le.classes_)}")

        for feat_name, feat_cols in feature_subsets.items():
            # Filter columns that exist
            feat_cols = [c for c in feat_cols if c in valid.columns]
            X = valid[feat_cols].values.astype(np.float32)
            print(f"\n  Running nested CV: {task}/{feat_name} "
                  f"(X shape: {X.shape})...")

            result = run_nested_cv(X, y, le, task, feat_name, n_classes)
            exp_name = f"XGB-{feat_name}" if feat_name != "full" else "XGB-full"
            if feat_name == "sentiment_only":
                exp_name = "XGB-sentiment-only"
            all_results[f"{task}_{exp_name}"] = result

    # ── PCA variant (addresses high p/n ratio: 254 features / 90 samples) ──
    print("\n  Running PCA-reduced XGBoost variant...")
    for task in config.TASKS:
        label_col = f"label_{task}"
        valid = merged[merged[label_col].notna()].copy()
        le = LabelEncoder()
        y = le.fit_transform(valid[label_col].values)
        n_classes = len(le.classes_)

        feat_cols = feature_subsets["full"]
        feat_cols = [c for c in feat_cols if c in valid.columns]
        X = valid[feat_cols].values.astype(np.float32)

        result = run_pca_nested_cv(X, y, le, task, n_classes)
        all_results[f"{task}_XGB-PCA"] = result

    # ── SHAP Analysis ────────────────────────────────────────────────────
    print("\n[3/4] SHAP analysis on best model...")
    shap_data = {}

    for task in config.TASKS:
        label_col = f"label_{task}"
        valid = merged[merged[label_col].notna()].copy()
        le = LabelEncoder()
        y = le.fit_transform(valid[label_col].values)
        n_classes = len(le.classes_)

        feat_cols = feature_subsets["full"]
        feat_cols = [c for c in feat_cols if c in valid.columns]
        X = valid[feat_cols].values.astype(np.float32)

        # Use best params from CV results
        cv_key = f"{task}_XGB-full"
        if cv_key in all_results:
            # Average best params across folds (use first fold's)
            best_params = all_results[cv_key]["fold_results"][0]["best_params"]
        else:
            best_params = {"max_depth": 4, "learning_rate": 0.1, "n_estimators": 100}

        model = train_final_model(X, y, best_params, n_classes)
        joblib.dump(model, config.MODELS_DIR / f"xgb_{task}_final.joblib")
        joblib.dump(le, config.MODELS_DIR / f"label_encoder_{task}.joblib")

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Normalise to a 3-D array (n_samples, n_features, n_classes) so all
        # downstream code is uniform regardless of SHAP version / binary vs
        # multiclass. Older SHAP returns a list of 2-D arrays; newer SHAP
        # returns a single array that is already 3-D for multiclass or 2-D for
        # binary.
        if isinstance(shap_values, list):
            sv_3d = np.stack(shap_values, axis=2)          # list → (N, F, C)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv_3d = shap_values                             # already (N, F, C)
        else:
            sv_3d = shap_values[:, :, np.newaxis]           # binary (N, F, 1)

        n_classes_shap = sv_3d.shape[2]

        # Beeswarm plot (show class 0 for simplicity; acceptable for overview)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(sv_3d[:, :, 0], X, feature_names=feat_cols,
                          show=False, max_display=20)
        plt.title(f"SHAP Summary — XGBoost {task.title()}")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"shap_beeswarm_{task}.png", dpi=150)
        plt.close()

        # Bar plot — mean |SHAP| across samples AND classes → always 1-D
        fig, ax = plt.subplots(figsize=(10, 8))
        mean_shap = np.abs(sv_3d).mean(axis=(0, 2))        # shape: (n_features,)
        top_idx = np.argsort(mean_shap)[-20:]               # 1-D integer array
        feat_cols_list = list(feat_cols)                    # ensure plain list
        plt.barh([feat_cols_list[int(i)] for i in top_idx], mean_shap[top_idx])
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Feature Importance — XGBoost {task.title()}")
        plt.tight_layout()
        plt.savefig(config.FIGURES_DIR / f"shap_bar_{task}.png", dpi=150)
        plt.close()

        # Per-sample top-K SHAP features (for GPT injection)
        sample_shap = {}
        tickers = valid.index.tolist()
        preds_all = model.predict(X)
        for i, ticker in enumerate(tickers):
            if n_classes_shap > 1:
                pred_class_idx = int(preds_all[i])
                sv = sv_3d[i, :, pred_class_idx]
            else:
                sv = sv_3d[i, :, 0]

            top_k_idx = np.argsort(np.abs(sv))[-config.SHAP_TOP_K:][::-1]
            sample_shap[ticker] = [
                (feat_cols_list[int(j)], float(sv[j])) for j in top_k_idx
            ]

        shap_data[task] = {
            "sample_shap": sample_shap,
            "shap_values": sv_3d,
        }

        # Save predictions for GPT-xgb-inject
        # IMPORTANT: use out-of-fold predictions from nested CV,
        # NOT model.predict(X) which is in-sample (data leakage).
        #
        # Both full-feature and sentiment-only OOF predictions are saved
        # as separate columns so 08_evaluation.py can evaluate each
        # variant independently without reusing the same prediction array.
        cv_result_full = all_results.get(f"{task}_XGB-full", {})
        oof_preds_full = np.array(cv_result_full.get("predictions", []))
        oof_probs_full = np.array(cv_result_full.get("probabilities", []))

        if len(oof_preds_full) == len(tickers) and len(oof_probs_full) == len(tickers):
            pred_labels_full = le.inverse_transform(oof_preds_full)
            confidences_full = oof_probs_full.max(axis=1)
            print(f"  Saving OOF predictions (full) for {task} ({len(pred_labels_full)} samples)")
        else:
            print(f"  WARNING: OOF predictions (full) not found for {task}, falling back to in-sample")
            in_preds = model.predict(X)
            in_probs = model.predict_proba(X)
            pred_labels_full = le.inverse_transform(in_preds)
            confidences_full = in_probs.max(axis=1)

        # Sentiment-only OOF predictions
        cv_result_sent = all_results.get(f"{task}_XGB-sentiment-only", {})
        oof_preds_sent = np.array(cv_result_sent.get("predictions", []))
        oof_probs_sent = np.array(cv_result_sent.get("probabilities", []))

        if len(oof_preds_sent) == len(tickers) and len(oof_probs_sent) == len(tickers):
            pred_labels_sent = le.inverse_transform(oof_preds_sent)
            confidences_sent = oof_probs_sent.max(axis=1)
            print(f"  Saving OOF predictions (sentiment-only) for {task} ({len(pred_labels_sent)} samples)")
        else:
            print(f"  WARNING: OOF predictions (sentiment-only) not found for {task}, skipping")
            pred_labels_sent = None
            confidences_sent = None

        preds_dict = {
            "symbol": tickers,
            f"xgb_pred_{task}": pred_labels_full,
            f"xgb_conf_{task}": confidences_full,
        }
        if pred_labels_sent is not None:
            preds_dict[f"xgb_pred_sentiment_only_{task}"] = pred_labels_sent
            preds_dict[f"xgb_conf_sentiment_only_{task}"] = confidences_sent

        xgb_preds = pd.DataFrame(preds_dict)
        xgb_preds.to_csv(config.RESULTS_DIR / f"xgb_predictions_{task}.csv", index=False)

    # Save SHAP data
    joblib.dump(shap_data, config.RESULTS_DIR / "shap_data.joblib")

    # ── Save all results ─────────────────────────────────────────────────
    print("\n[4/4] Saving results...")

    # Convert numpy arrays for JSON serialization
    serializable_results = {}
    for k, v in all_results.items():
        sv = dict(v)
        sv.pop("predictions", None)
        sv.pop("probabilities", None)
        serializable_results[k] = sv

    with open(config.RESULTS_DIR / "xgb_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("XGBOOST BASELINE COMPLETE")
    print(f"\n  Results summary:")
    for k, v in all_results.items():
        print(f"    {k}: F1={v['f1_macro']:.4f}, Acc={v['accuracy']:.4f}")
    print(f"\n  Models saved to: {config.MODELS_DIR}")
    print(f"  SHAP plots saved to: {config.FIGURES_DIR}")
    print(f"  Predictions saved to: {config.RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()