"""Optional supervised 'mine vs other' style classifier (LogisticRegression default)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from ..features.surface_features import SurfaceFeatureExtractor, SURFACE_FEATURE_NAMES
from ..io import ensure_parent


@dataclass
class SupervisedStyleModel:
    classifier: LogisticRegression
    feature_names: list[str]
    cv_accuracy: float | None

    def predict_proba_mine(self, texts: list[str]) -> np.ndarray:
        X = SurfaceFeatureExtractor().fit_transform(texts)
        idx = list(self.classifier.classes_).index(1)  # 'mine' == class 1
        return self.classifier.predict_proba(X)[:, idx]


def _read_labeled_csv(path: str | Path, text_col: str, label_col: str):
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or text_col not in reader.fieldnames \
                or label_col not in reader.fieldnames:
            raise RuntimeError(f"CSV must have '{text_col}' and '{label_col}' columns")
        for i, row in enumerate(reader):
            text = (row.get(text_col) or "").strip()
            label = (row.get(label_col) or "").strip()
            if not text or not label:
                continue
            texts.append(text)
            labels.append(1 if label == "mine" else 0)
            groups.append(row.get("source") or f"row{i}")
    return texts, labels, groups


def train_supervised(csv_path: str | Path, *, text_col: str = "text",
                     label_col: str = "label") -> SupervisedStyleModel:
    texts, labels, groups = _read_labeled_csv(csv_path, text_col, label_col)
    if len(set(labels)) < 2:
        raise RuntimeError("Supervised training needs both 'mine' and 'other' labels.")
    X = SurfaceFeatureExtractor().fit_transform(texts)
    y = np.asarray(labels)

    cv_accuracy: float | None = None
    n_groups = len(set(groups))
    if n_groups >= 2 and len(y) >= 4:
        n_splits = min(n_groups, 5)
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=1000), X, y,
                groups=groups, cv=GroupKFold(n_splits=n_splits))
            cv_accuracy = float(scores.mean())
        except Exception:
            cv_accuracy = None

    classifier = LogisticRegression(max_iter=1000).fit(X, y)
    return SupervisedStyleModel(
        classifier=classifier, feature_names=list(SURFACE_FEATURE_NAMES),
        cv_accuracy=cv_accuracy)


def save_model(model: SupervisedStyleModel, path: str | Path) -> None:
    joblib.dump(model, ensure_parent(path))
