"""Logistic classifier using lagged public observations only."""

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .market import FEATURES


class Detector:
    def __init__(
        self,
        mean,
        scale,
        coefficients,
        intercept,
    ):
        self.mean = np.array(mean)
        self.scale = np.array(scale)
        self.coefficients = np.array(coefficients)
        self.intercept = float(intercept)

    @classmethod
    def fit(cls, tapes):
        x = np.concatenate([
            tape[FEATURES].to_numpy()
            for tape in tapes
        ])

        y = np.concatenate([
            tape["informed"].to_numpy()
            for tape in tapes
        ])

        # Fit preprocessing exclusively on training sessions.
        scaler = StandardScaler().fit(x)

        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=0,
        ).fit(scaler.transform(x), y)

        return cls(
            scaler.mean_,
            scaler.scale_,
            model.coef_[0],
            model.intercept_[0],
        )

    def predict(self, tape):
        x = tape[FEATURES].to_numpy(dtype=float)
        standardized = (x - self.mean) / self.scale

        score = (
            standardized @ self.coefficients
            + self.intercept
        )

        return 1 / (
            1 + np.exp(-np.clip(score, -40, 40))
        )

    def save(self, path):
        data = {
            "features": FEATURES,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

        path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text(encoding="utf-8"))

        if data.pop("features") != FEATURES:
            raise ValueError("Feature order mismatch")

        return cls(**data)


def metrics(tapes, detector):
    y = np.concatenate([
        tape["informed"].to_numpy()
        for tape in tapes
    ])

    probabilities = np.concatenate([
        detector.predict(tape)
        for tape in tapes
    ])

    has_both_classes = len(np.unique(y)) == 2

    return {
        "arrivals": len(y),
        "informed_rate": float(y.mean()),
        "roc_auc": (
            float(roc_auc_score(y, probabilities))
            if has_both_classes
            else None
        ),
        "average_precision": (
            float(average_precision_score(y, probabilities))
            if y.sum()
            else None
        ),
        "brier": float(
            brier_score_loss(y, probabilities)
        ),
        "log_loss": float(
            log_loss(y, probabilities, labels=[0, 1])
        ),
    }
