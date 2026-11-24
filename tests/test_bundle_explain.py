import numpy as np
import pytest

from blunder_radar.explain.explainer import explain_row, summarise, tree_contributions


def test_calibrated_probabilities_are_valid(tiny_bundle, tiny_features):
    p = tiny_bundle.predict_proba(tiny_features)
    assert p.shape == (len(tiny_features),)
    assert np.all((p >= 0) & (p <= 1))


def test_bundle_rejects_missing_columns(tiny_bundle, tiny_features):
    with pytest.raises(KeyError):
        tiny_bundle.predict_proba(tiny_features.drop(columns=["mover_mobility"]))


def test_risk_bands_are_ordered(tiny_bundle):
    base = tiny_bundle.base_rate
    bands = [tiny_bundle.risk_band(p) for p in (0.1 * base, base, 3 * base, 10 * base)]
    assert bands == ["low", "moderate", "high", "very high"]


def test_bundle_roundtrip(tiny_bundle, tiny_bundle_path, tiny_features):
    from blunder_radar.modeling.bundle import ModelBundle

    loaded = ModelBundle.load(tiny_bundle_path)
    a = tiny_bundle.predict_proba(tiny_features.head(50))
    b = loaded.predict_proba(tiny_features.head(50))
    assert np.allclose(a, b)


def test_tree_contributions_are_additive(tiny_bundle, tiny_features):
    """TreeSHAP values + bias must reproduce the model's log-odds exactly."""
    X = tiny_features.head(25)
    contribs, _ = tree_contributions(tiny_bundle, X)
    import xgboost as xgb

    full = tiny_bundle.model.get_booster().predict(
        xgb.DMatrix(X[tiny_bundle.feature_names]), pred_contribs=True
    )
    margin = tiny_bundle.model.get_booster().predict(
        xgb.DMatrix(X[tiny_bundle.feature_names]), output_margin=True
    )
    assert np.allclose(full.sum(axis=1), margin, atol=1e-4)
    assert contribs.shape == (25, len(tiny_bundle.feature_names))


def test_explain_row_returns_ranked_factors(tiny_bundle, tiny_features):
    factors = explain_row(tiny_bundle, tiny_features.head(1), top_k=4)
    assert len(factors) == 4
    strengths = [abs(f["contribution"]) for f in factors]
    assert strengths == sorted(strengths, reverse=True)
    assert all(f["direction"] in ("raises risk", "lowers risk") for f in factors)
    assert all(f["text"].startswith(f["direction"]) for f in factors)


def test_summary_mentions_band_and_ratio():
    text = summarise(0.2, "high", 0.1, [
        {"direction": "raises risk", "text": "raises risk: only 9s left on the clock"}
    ])
    assert "High blunder risk" in text and "2.0x" in text and "9s left" in text


def test_lightgbm_contributions_are_additive(tiny_features):
    from lightgbm import LGBMClassifier

    from blunder_radar.modeling.bundle import ModelBundle
    from blunder_radar.modeling.dataset import feature_columns

    cols = feature_columns(with_eval=False)
    model = LGBMClassifier(n_estimators=30, num_leaves=8, n_jobs=1, verbose=-1, random_state=0)
    model.fit(tiny_features[cols], tiny_features["is_blunder"])
    bundle = ModelBundle(model=model, feature_names=cols, algorithm="lightgbm", with_eval=False)
    X = tiny_features.head(40)
    contribs, bias = tree_contributions(bundle, X)
    raw = bundle.predict_raw(X)
    assert np.allclose(contribs.sum(axis=1) + bias, np.log(raw / (1 - raw)), atol=1e-6)


def test_every_feature_has_a_plain_english_phrase():
    from blunder_radar.explain.explainer import PHRASES
    from blunder_radar.modeling.dataset import feature_columns

    assert [f for f in feature_columns(with_eval=True) if f not in PHRASES] == []
