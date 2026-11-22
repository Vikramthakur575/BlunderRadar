"""
Phase 3: model training, calibration, comparison and experiment tracking.

Experiment grid
  feature modes : without_eval (deployable live) and with_eval (uses pre-move engine eval)
  algorithms    : majority class, logistic regression, XGBoost, LightGBM
  protocol      : chronological, game-level train / val / test split (no game in two splits),
                  Optuna tuning on a train subsample scored on validation log loss,
                  probability calibration fitted on the validation split only.

Everything is tracked in MLflow and summarised in docs/RESULTS.md.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from blunder_radar.config import ModelConfig, load_model_config  # noqa: E402
from blunder_radar.explain.explainer import tree_contributions  # noqa: E402
from blunder_radar.modeling.bundle import ModelBundle, default_thresholds  # noqa: E402
from blunder_radar.modeling.dataset import (  # noqa: E402
    TARGET_COL,
    assert_no_game_leakage,
    chronological_game_split,
    feature_columns,
    load_features,
)
from blunder_radar.modeling.metrics import calibration_points, evaluate  # noqa: E402

logger = logging.getLogger("blunder_radar.train")

TREE_ALGOS = ("xgboost", "lightgbm")
ALL_ALGOS = ("majority_class", "logistic_regression", "xgboost", "lightgbm")


# --------------------------------------------------------------------------- calibration
class PlattCalibrator:
    """Sigmoid (Platt) calibration of raw probabilities via logistic regression on logits."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1e6, max_iter=500)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        self._lr.fit(self._logit(raw), y)
        return self

    def predict(self, raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(self._logit(raw))[:, 1]


def fit_calibrator(raw: np.ndarray, y: np.ndarray, method: str = "isotonic"):
    """Fit a calibrator on held-out (validation) predictions."""
    if method == "sigmoid":
        return PlattCalibrator().fit(raw, y)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.fit(raw, y)
    return iso


# --------------------------------------------------------------------------- estimators
def make_estimator(algorithm: str, params: dict[str, Any] | None = None, seed: int = 42):
    params = dict(params or {})
    if algorithm == "majority_class":
        return DummyClassifier(strategy="prior")
    if algorithm == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=params.get("C", 1.0), max_iter=500),
        )
    if algorithm == "xgboost":
        from xgboost import XGBClassifier

        base = {
            "n_estimators": 400,
            "learning_rate": 0.08,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        }
        base.update(params)
        return XGBClassifier(
            **base,
            tree_method="hist",
            eval_metric="logloss",
            early_stopping_rounds=30,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    if algorithm == "lightgbm":
        from lightgbm import LGBMClassifier

        base = {
            "n_estimators": 400,
            "learning_rate": 0.08,
            "num_leaves": 63,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
        }
        base.update(params)
        return LGBMClassifier(**base, n_jobs=1, random_state=seed, verbose=-1)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def fit_estimator(algorithm, params, X_tr, y_tr, X_va, y_va, seed: int = 42):
    """Fit with early stopping on the validation split where the algorithm supports it."""
    est = make_estimator(algorithm, params, seed)
    if algorithm == "xgboost":
        est.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    elif algorithm == "lightgbm":
        import lightgbm as lgb

        est.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
    else:
        est.fit(X_tr, y_tr)
    return est


def tune_hyperparameters(
    algorithm: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    n_trials: int,
    timeout: float,
    seed: int,
    sample_size: int,
) -> dict[str, Any]:
    """Optuna TPE search minimising validation log loss on a training subsample."""
    import optuna
    from sklearn.metrics import log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if len(X_tr) > sample_size:
        idx = np.random.default_rng(seed).choice(len(X_tr), size=sample_size, replace=False)
        X_s, y_s = X_tr.iloc[idx], y_tr.iloc[idx]
    else:
        X_s, y_s = X_tr, y_tr

    def objective(trial: optuna.Trial) -> float:
        if algorithm == "xgboost":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "n_estimators": 250,
            }
        else:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "n_estimators": 250,
            }
        est = fit_estimator(algorithm, params, X_s, y_s, X_va, y_va, seed)
        p = np.clip(est.predict_proba(X_va)[:, 1], 1e-6, 1 - 1e-6)
        return float(log_loss(y_va, p))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    best = dict(study.best_params) if len(study.trials) else {}
    best.pop("n_estimators", None)
    return best


# --------------------------------------------------------------------------- experiment
@dataclass
class FittedRun:
    algorithm: str
    mode: str
    params: dict[str, Any]
    model: Any
    calibrator: Any
    metrics: dict[str, float]
    feature_names: list[str]


def _prefixed(prefix: str, m: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in m.items()}


def _mlflow_setup(cfg: ModelConfig):
    import mlflow

    uri = cfg.mlflow_tracking_uri
    if "://" not in uri and not uri.startswith("sqlite:"):
        # A plain directory name: keep runs in a SQLite store inside it (the file-store
        # backend is deprecated in current MLflow releases).
        directory = Path(uri).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        uri = f"sqlite:///{directory / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.experiment_name)
    return mlflow


def reliability_plot(y_true, raw, cal, path: Path, title: str) -> None:
    """Reliability diagram (quantile bins) before and after calibration."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    hi = 0.0
    for name, p, style in (("raw", raw, "o--"), ("calibrated", cal, "s-")):
        pred, obs, _ = calibration_points(y_true, p, n_bins=10)
        ax.plot(pred, obs, style, label=name)
        hi = max(hi, max(pred), max(obs))
    hi = min(1.0, hi * 1.08 + 0.01)
    ax.plot([0, hi], [0, hi], ":", color="grey", label="perfect")
    ax.set_xlabel("Mean predicted blunder probability")
    ax.set_ylabel("Observed blunder rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def importance_plot(bundle: ModelBundle, X: pd.DataFrame, path: Path, top: int = 15) -> list[dict]:
    """Mean |TreeSHAP| per feature (log-odds) bar chart; returns the ranking."""
    contribs, _ = tree_contributions(bundle, X)
    mean_abs = np.abs(contribs).mean(axis=0)
    order = np.argsort(-mean_abs)[:top]
    ranking = [
        {"feature": bundle.feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order
    ]
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.barh([r["feature"] for r in ranking][::-1], [r["mean_abs_shap"] for r in ranking][::-1])
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title("What drives blunder risk (global importance)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    try:
        import shap

        n = min(2000, len(X))
        sample = X.iloc[:n][bundle.feature_names]
        plt.figure(figsize=(8, 6))
        shap.summary_plot(contribs[:n], sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(path.with_name("shap_summary.png"), dpi=140, bbox_inches="tight")
        plt.close("all")
    except Exception as exc:  # plotting is best-effort
        logger.warning("SHAP summary plot skipped: %s", exc)
    return ranking


def train_all(
    cfg: ModelConfig | None = None,
    features_path: str | Path | None = None,
    models_dir: str | Path | None = None,
    docs_dir: str | Path | None = None,
    quick: bool = False,
    n_trials: int | None = None,
    algorithms: tuple[str, ...] = ALL_ALGOS,
    max_games: int | None = None,
    write_docs: bool = True,
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Run the full experiment grid and persist the deployable model bundles."""
    cfg = cfg or load_model_config()
    features_path = Path(features_path or cfg.features_path)
    models_dir = Path(models_dir or cfg.models_dir)
    docs_dir = Path(docs_dir or cfg.docs_dir)
    seed = cfg.random_seed
    trials = n_trials if n_trials is not None else (1 if quick else cfg.optuna_trials)
    t_start = time.time()

    df = load_features(features_path)
    if max_games:
        keep = df["game_id"].drop_duplicates().iloc[:max_games]
        df = df[df["game_id"].isin(keep)].reset_index(drop=True)

    split = chronological_game_split(df, cfg.val_size, cfg.test_size)
    assert_no_game_leakage(df, split)
    y = df[TARGET_COL].astype(int)
    masks = {s: (split == s).to_numpy() for s in ("train", "val", "test")}
    sizes = {s: int(m.sum()) for s, m in masks.items()}
    game_counts = {s: int(df.loc[m, "game_id"].nunique()) for s, m in masks.items()}
    base_rates = {s: float(y[m].mean()) for s, m in masks.items()}
    logger.info("Split sizes: %s | games: %s | base rates: %s", sizes, game_counts, base_rates)

    mlflow = _mlflow_setup(cfg) if use_mlflow else None
    runs: list[FittedRun] = []
    assets = docs_dir / "assets"

    for mode in ("without_eval", "with_eval"):
        cols = feature_columns(with_eval=(mode == "with_eval"))
        X = df[cols]
        X_tr, y_tr = X[masks["train"]], y[masks["train"]]
        X_va, y_va = X[masks["val"]], y[masks["val"]]
        X_te, y_te = X[masks["test"]], y[masks["test"]]

        for algo in algorithms:
            t0 = time.time()
            params: dict[str, Any] = {}
            if algo in TREE_ALGOS and trials > 0:
                params = tune_hyperparameters(
                    algo, X_tr, y_tr, X_va, y_va, trials, cfg.optuna_timeout, seed,
                    cfg.tune_sample_size,
                )
            model = fit_estimator(algo, params, X_tr, y_tr, X_va, y_va, seed)
            raw_va = model.predict_proba(X_va)[:, 1]
            raw_te = model.predict_proba(X_te)[:, 1]
            cal = fit_calibrator(raw_va, y_va.to_numpy(), cfg.calibration_method)
            cal_te = cal.predict(raw_te)
            metrics = {
                **_prefixed("val_raw", evaluate(y_va, raw_va)),
                **_prefixed("test_raw", evaluate(y_te, raw_te)),
                **_prefixed("test_cal", evaluate(y_te, cal_te)),
                "fit_seconds": round(time.time() - t0, 1),
            }
            run = FittedRun(algo, mode, params, model, cal, metrics, cols)
            runs.append(run)
            logger.info(
                "[%s/%s] test PR-AUC %.4f  logloss raw %.4f -> cal %.4f  (%.0fs)",
                mode, algo, metrics["test_raw_pr_auc"], metrics["test_raw_log_loss"],
                metrics["test_cal_log_loss"], metrics["fit_seconds"],
            )
            if mlflow is not None:
                with mlflow.start_run(run_name=f"{algo}-{mode}"):
                    mlflow.log_params({"algorithm": algo, "mode": mode, "n_features": len(cols),
                                       "calibration": cfg.calibration_method, **params})
                    mlflow.log_metrics({k: v for k, v in metrics.items() if np.isfinite(v)})

    # ---- choose the deployable model per mode: lowest VALIDATION raw log loss ----
    def pick(mode: str) -> FittedRun:
        pool = [r for r in runs if r.mode == mode and r.algorithm != "majority_class"]
        return min(pool, key=lambda r: r.metrics["val_raw_log_loss"])

    deployed: dict[str, ModelBundle] = {}
    ranking: list[dict] = []
    for mode in ("without_eval", "with_eval"):
        r = pick(mode)
        base = base_rates["train"]
        bundle = ModelBundle(
            model=r.model,
            feature_names=r.feature_names,
            algorithm=r.algorithm,
            with_eval=(mode == "with_eval"),
            calibrator=r.calibrator,
            base_rate=base,
            risk_thresholds=default_thresholds(base),
            metrics=r.metrics,
            meta={
                "mode": mode,
                "params": r.params,
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "split_sizes": sizes,
                "split_games": game_counts,
                "split_base_rates": base_rates,
                "calibration": cfg.calibration_method,
                "data_time_span": [str(df["game_timestamp"].min()), str(df["game_timestamp"].max())],
                "selection_rule": "lowest validation raw log loss (non-baseline models)",
            },
        )
        deployed[mode] = bundle
        name = "best_model.joblib" if mode == "without_eval" else "best_model_with_eval.joblib"
        bundle.save(models_dir / name)

    live = deployed["without_eval"]
    X_test_live = df.loc[masks["test"], live.feature_names]
    y_test = y[masks["test"]]
    raw_live = live.predict_raw(X_test_live)
    cal_live = live.predict_proba(X_test_live)
    reliability_plot(
        y_test, raw_live, cal_live, assets / "calibration_live_model.png",
        f"Calibration on held-out games\n({live.algorithm}, no engine evaluation)",
    )
    ranking = importance_plot(live, X_test_live.iloc[:5000], assets / "feature_importance.png")
    live.meta["top_features"] = ranking[:10]
    live.save(models_dir / "best_model.joblib")

    # ---- per-rating-band models vs one global model (deployed config, no engine eval) ----
    band_rows: list[dict[str, Any]] = []
    r_live = pick("without_eval")
    for band in ("low", "mid", "high"):
        in_band = (df["rating_band"] == band).to_numpy()
        tr, va, te = (masks[s] & in_band for s in ("train", "val", "test"))
        if tr.sum() < 500 or va.sum() < 100 or te.sum() < 100 or y[tr].nunique() < 2:
            continue
        cols = live.feature_names
        m_band = fit_estimator(r_live.algorithm, r_live.params, df.loc[tr, cols], y[tr],
                               df.loc[va, cols], y[va], seed)
        cal_b = fit_calibrator(m_band.predict_proba(df.loc[va, cols])[:, 1], y[va].to_numpy(),
                               cfg.calibration_method)
        p_band = cal_b.predict(m_band.predict_proba(df.loc[te, cols])[:, 1])
        p_glob = live.predict_proba(df.loc[te, cols])
        mb, mg = evaluate(y[te], p_band), evaluate(y[te], p_glob)
        band_rows.append({
            "band": band, "test_positions": int(te.sum()), "test_blunders": int(y[te].sum()),
            "global_pr_auc": mg["pr_auc"], "band_pr_auc": mb["pr_auc"],
            "global_log_loss": mg["log_loss"], "band_log_loss": mb["log_loss"],
        })

    # ---- persist results ----
    table = pd.DataFrame(
        [
            {"mode": r.mode, "algorithm": r.algorithm, **r.metrics, "params": json.dumps(r.params)}
            for r in runs
        ]
    )
    models_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    table.to_csv(assets / "experiment_results.csv", index=False)
    summary = {
        "sizes": sizes, "games": game_counts, "base_rates": base_rates,
        "deployed": {m: {"algorithm": b.algorithm, "params": b.meta["params"]}
                     for m, b in deployed.items()},
        "per_band": band_rows,
        "top_features": ranking,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    (models_dir / "model_card.json").write_text(json.dumps(summary, indent=2, default=float))

    if mlflow is not None:
        try:
            with mlflow.start_run(run_name="deployed-model"):
                mlflow.log_params({"algorithm": live.algorithm, **live.meta["params"]})
                mlflow.log_metrics({k: v for k, v in live.metrics.items() if np.isfinite(v)})
                mlflow.log_artifact(str(models_dir / "best_model.joblib"))
                mlflow.log_artifact(str(assets / "calibration_live_model.png"))
                try:
                    mlflow.sklearn.log_model(live.model, name="model",
                                             registered_model_name="blunder-radar")
                except Exception as exc:  # registry support depends on the backend store
                    logger.warning("MLflow model registration skipped: %s", exc)
        except Exception as exc:
            logger.warning("MLflow final logging skipped: %s", exc)

    if write_docs:
        write_results_markdown(docs_dir / "RESULTS.md", table, summary)
    logger.info("Training finished in %.0fs", time.time() - t_start)
    return {"table": table, "summary": summary, "bundles": deployed}


def write_results_markdown(path: Path, table: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Auto-generated, honest results report."""

    def row_for(mode: str, algo: str) -> pd.Series:
        return table[(table["mode"] == mode) & (table["algorithm"] == algo)].iloc[0]

    lines: list[str] = ["# BlunderRadar: Results", "",
                        "_Auto-generated by `blunder-radar model train`. Do not edit by hand._", ""]
    s = summary
    lines += ["## 1. Data and protocol", "",
              "| Split | Positions | Games | Blunder rate |", "| :-- | --: | --: | --: |"]
    for k in ("train", "val", "test"):
        lines.append(f"| {k} | {s['sizes'][k]:,} | {s['games'][k]:,} | {s['base_rates'][k]*100:.2f}% |")
    lines += ["",
              "Splits are **chronological and game-level**: games are ordered by start time, "
              "so the model is always tested on games played after the ones it was trained on, "
              "and no game appears in two splits. Calibration is fitted on the validation split only.", ""]

    lines += ["## 2. Model comparison (held-out test games)", "",
              "Metrics on the test split after calibration. PR-AUC is the headline metric for this "
              "rare-event problem; compare it with the base rate (the PR-AUC of a random model).", "",
              "| Features | Algorithm | PR-AUC | ROC-AUC | Log loss | Brier | ECE |",
              "| :-- | :-- | --: | --: | --: | --: | --: |"]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['mode']} | {r['algorithm']} | {r['test_cal_pr_auc']:.4f} | {r['test_cal_roc_auc']:.4f} | "
            f"{r['test_cal_log_loss']:.4f} | {r['test_cal_brier']:.4f} | {r['test_cal_ece']:.4f} |"
        )
    dep = s["deployed"]
    lines += ["", f"**Deployed (live) model:** `{dep['without_eval']['algorithm']}` without engine "
              f"evaluation, chosen by lowest validation log loss. Best model that may use the "
              f"pre-move engine evaluation: `{dep['with_eval']['algorithm']}`.", ""]

    a_wo = row_for("without_eval", dep["without_eval"]["algorithm"])
    a_w = row_for("with_eval", dep["with_eval"]["algorithm"])
    gain = (a_w["test_cal_pr_auc"] - a_wo["test_cal_pr_auc"]) / max(a_wo["test_cal_pr_auc"], 1e-9)
    lines += ["## 3. Is `eval_before` doing most of the work?", "",
              f"Adding the engine's pre-move evaluation changes test PR-AUC from "
              f"**{a_wo['test_cal_pr_auc']:.4f}** to **{a_w['test_cal_pr_auc']:.4f}** "
              f"({gain*100:+.1f}%), and log loss from {a_wo['test_cal_log_loss']:.4f} to "
              f"{a_w['test_cal_log_loss']:.4f}. ", ""]
    if gain > 0.25:
        lines.append("The engine evaluation carries a large share of the signal. A model that needs it "
                     "is not usable in a live setting without running an engine, which is why the "
                     "deployed model excludes it.")
    elif gain > 0.05:
        lines.append("The engine evaluation helps noticeably but the board and clock features carry "
                     "most of the signal, so the deployed model (which excludes it) is still useful.")
    else:
        lines.append("The engine evaluation adds little beyond the board and clock features.")
    lines += [""]

    lines += ["## 4. Calibration", "",
              "![Reliability diagram](assets/calibration_live_model.png)", "",
              f"Calibrated test ECE for the deployed model: **{a_wo['test_cal_ece']:.4f}** "
              f"(raw: {a_wo['test_raw_ece']:.4f}); log loss {a_wo['test_raw_log_loss']:.4f} raw "
              f"-> {a_wo['test_cal_log_loss']:.4f} calibrated.", ""]

    if s["per_band"]:
        lines += ["## 5. Per-rating-band models vs one global model", "",
                  "| Band | Test positions | Blunders | Global PR-AUC | Band PR-AUC | Global log loss | Band log loss |",
                  "| :-- | --: | --: | --: | --: | --: | --: |"]
        for b in s["per_band"]:
            lines.append(
                f"| {b['band']} | {b['test_positions']:,} | {b['test_blunders']:,} | "
                f"{b['global_pr_auc']:.4f} | {b['band_pr_auc']:.4f} | "
                f"{b['global_log_loss']:.4f} | {b['band_log_loss']:.4f} |"
            )
        wins = sum(1 for b in s["per_band"] if b["band_pr_auc"] > b["global_pr_auc"])
        lines += ["", f"Band-specific models beat the global model on PR-AUC in {wins} of "
                  f"{len(s['per_band'])} bands. Rating is already a model input, so the global model "
                  "shares data across bands; it is the one deployed.", ""]

    lines += ["## 6. What drives the predictions", "",
              "![Feature importance](assets/feature_importance.png)", "",
              "![SHAP summary](assets/shap_summary.png)", "",
              "Top features by mean |SHAP|: "
              + ", ".join(f"`{f['feature']}`" for f in s["top_features"][:8]) + ".", ""]

    lines += ["## 7. Limitations (read before trusting these numbers)", "",
              "- All games come from a short window of one day, so there is no seasonal or "
              "population drift test. The chronological split is real but narrow.",
              "- Lichess only publishes engine evaluations for games a player requested analysis "
              "for, so the data is a **selected** sample, not all games.",
              "- Positions are correlated within a game; this is why splits are made per game.",
              "- Blunder labels depend on the engine's evaluation and the chosen threshold "
              "(200 centipawns here).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
