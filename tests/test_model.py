import numpy as np

from src.model import Scorer, cost_threshold


def test_cost_threshold_drops_when_false_negatives_are_costly():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(950), np.ones(50)].astype(int)
    proba = np.r_[rng.uniform(0.0, 0.6, 950), rng.uniform(0.4, 1.0, 50)]
    t_cheap, _ = cost_threshold(y, proba, cost_fp=1, cost_fn=1)
    t_costly, _ = cost_threshold(y, proba, cost_fp=1, cost_fn=100)
    assert t_costly <= t_cheap  # costly misses -> catch more -> lower threshold


def test_scorer_applies_threshold():
    class FakeModel:
        def predict_proba(self, x):
            return np.array([[0.2, 0.8]])

    s = Scorer(FakeModel(), threshold=0.5, feature_names=["a", "b"], version="v1")
    out = s.score_one({"a": 1.0, "b": 2.0})
    assert out["is_fraud"] is True
    assert out["probability"] == 0.8
    assert out["model_version"] == "v1"
