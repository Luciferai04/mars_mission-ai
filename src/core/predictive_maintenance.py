#!/usr/bin/env python3
"""
Predictive Maintenance module using simple ML (IsolationForest / Logistic Regression).

Provides:
- MaintenancePredictor: fit, predict_proba, save, load
- Feature extraction from telemetry dicts
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np

try:  # Optional sklearn import
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
except Exception:
    IsolationForest = None  # type: ignore
    LogisticRegression = None  # type: ignore
    StandardScaler = None  # type: ignore
    Pipeline = None  # type: ignore
    joblib = None  # type: ignore


FEATURE_KEYS = [
    "battery_soc",
    "power_generation",
    "power_consumption",
    "temp_c",
    "wind_speed",
    "pressure",
    "uv_index",
]


def _extract_features(telemetry: Dict[str, Any]) -> np.ndarray:
    vals = []
    for k in FEATURE_KEYS:
        v = telemetry.get(k, 0.0)
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


class MaintenancePredictor:
    def __init__(self) -> None:
        if Pipeline is None or LogisticRegression is None or StandardScaler is None:
            raise ImportError("scikit-learn is required for MaintenancePredictor")
        self.model: Pipeline = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500))]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_proba(self, telemetry: Dict[str, Any]) -> float:
        x = _extract_features(telemetry).reshape(1, -1)
        proba = self.model.predict_proba(x)[0, 1]
        return float(proba)

    def save(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib not available")
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        if joblib is None:
            raise ImportError("joblib not available")
        self.model = joblib.load(path)
