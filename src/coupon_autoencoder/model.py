"""Fit reconstruction models on normal training data; calibrate on later normal data."""

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, log_features


def calibrate(scores, alert_rate=0.01):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be a nonempty finite vector")
    if not 0 < alert_rate < 1:
        raise ValueError("Alert rate must be between zero and one")
    return float(np.quantile(scores, 1 - alert_rate, method="higher"))


class Detector:
    def __init__(self, seed=42, alert_rate=0.01):
        self.seed = seed
        self.alert_rate = alert_rate

    def fit(self, train_normal, validation_normal):
        # Neither labels nor test data are accepted by this interface.
        self.scaler_ = StandardScaler().fit(log_features(train_normal))
        train = self.scaler_.transform(log_features(train_normal))
        self.autoencoder_ = MLPRegressor(
            hidden_layer_sizes=(24, 6, 24), activation="tanh", solver="adam",
            alpha=0.001, batch_size=128, learning_rate_init=0.002,
            max_iter=250, n_iter_no_change=25, tol=1e-5,
            early_stopping=False, random_state=self.seed,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            self.autoencoder_.fit(train, train)
        self.training_warnings_ = [str(w.message) for w in captured]
        self.pca_ = PCA(n_components=6, svd_solver="full").fit(train)
        self.isolation_ = IsolationForest(
            n_estimators=150, max_samples=min(256, len(train)),
            random_state=self.seed, n_jobs=1,
        ).fit(train)
        validation = self.scores(validation_normal)
        self.thresholds_ = {name: calibrate(s, self.alert_rate) for name, s in validation.items()}
        return self

    def scores(self, x):
        z = self.scaler_.transform(log_features(x))
        reconstruction = self.autoencoder_.predict(z)
        pca_reconstruction = self.pca_.inverse_transform(self.pca_.transform(z))
        return {
            "autoencoder": np.mean((z - reconstruction) ** 2, axis=1),
            "pca": np.mean((z - pca_reconstruction) ** 2, axis=1),
            "isolation_forest": -self.isolation_.score_samples(z),
            "max_abs_z_rule": np.max(np.abs(z), axis=1),
        }

    def contributions(self, x):
        z = self.scaler_.transform(log_features(x))
        return (z - self.autoencoder_.predict(z)) ** 2

    def bundle(self):
        return {
            "schema_version": 1, "features": list(FEATURES),
            "activation": "tanh", "mean": self.scaler_.mean_.tolist(),
            "scale": self.scaler_.scale_.tolist(),
            "weights": [w.tolist() for w in self.autoencoder_.coefs_],
            "biases": [b.tolist() for b in self.autoencoder_.intercepts_],
            "threshold": self.thresholds_["autoencoder"],
            "threshold_source": "later_normal_validation_only",
            "seed": self.seed, "alert_rate": self.alert_rate,
        }


def save_bundle(detector, path):
    Path(path).write_text(json.dumps(detector.bundle(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def score_bundle(bundle, x):
    # Portable JSON + NumPy inference, no pickle or joblib deserialization.
    if bundle.get("schema_version") != 1 or bundle.get("features") != list(FEATURES):
        raise ValueError("Unsupported model schema or feature order")
    if bundle.get("activation") != "tanh":
        raise ValueError("Unsupported hidden activation")
    mean, scale = np.asarray(bundle["mean"]), np.asarray(bundle["scale"])
    if mean.shape != (12,) or scale.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid scaling parameters")
    z = (log_features(x) - mean) / scale
    output = z
    weights, biases = bundle["weights"], bundle["biases"]
    if len(weights) != 4 or len(biases) != 4:
        raise ValueError("Expected four weight matrices and four biases")
    dimensions = (12, 24, 6, 24, 12)
    for i, (w, b) in enumerate(zip(weights, biases)):
        w, b = np.asarray(w), np.asarray(b)
        if w.shape != (dimensions[i], dimensions[i+1]) or b.shape != (dimensions[i+1],):
            raise ValueError("Invalid layer dimensions")
        if not np.isfinite(w).all() or not np.isfinite(b).all():
            raise ValueError("Model parameters must be finite")
        output = output @ w + b
        if i < len(weights) - 1:
            output = np.tanh(output)
    threshold = float(bundle["threshold"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid threshold")
    scores = np.mean((z - output) ** 2, axis=1)
    return scores, scores > threshold
