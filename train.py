"""Train the category + priority classifiers and persist them with joblib.

    python train.py                 # train on the 80% split, report on the 20% test split
    python train.py --regenerate    # also rebuild data/tickets.csv from the generator
    python train.py --refit-full    # after evaluating, refit on 100% of the data before saving
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.classifier import make_text
from app.dataset import DATA_PATH, load_csv, write_csv

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
METRICS_PATH = MODEL_PATH.with_name("metrics.json")
SEED = 42


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("svc", LinearSVC(C=1.0, class_weight="balanced", random_state=SEED)),
    ])


def evaluate(name: str, x_train, x_test, y_train, y_test) -> dict:
    model = build_pipeline().fit(x_train, y_train)
    pred = model.predict(x_test)
    cv = cross_val_score(build_pipeline(), x_train, y_train, cv=5, scoring="accuracy")
    return {
        "target": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "macro_f1": round(f1_score(y_test, pred, average="macro"), 4),
        "cv5_accuracy_mean": round(float(cv.mean()), 4),
        "cv5_accuracy_std": round(float(cv.std()), 4),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }


def main(regenerate: bool = False, refit_full: bool = False) -> dict:
    rows = write_csv() if regenerate else load_csv()
    texts = [make_text(r["subject"], r["body"]) for r in rows]
    cats = [r["category"] for r in rows]
    pris = [r["priority"] for r in rows]

    idx = list(range(len(rows)))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=cats)
    pick = lambda arr, ids: [arr[i] for i in ids]  # noqa: E731
    xtr, xte = pick(texts, train_idx), pick(texts, test_idx)

    metrics = {
        "dataset_rows": len(rows),
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "sklearn_version": sklearn.__version__,
        "refit_full": refit_full,
        "category": evaluate("category", xtr, xte, pick(cats, train_idx), pick(cats, test_idx)),
        "priority": evaluate("priority", xtr, xte, pick(pris, train_idx), pick(pris, test_idx)),
    }

    if refit_full:
        xtr, ytr_c, ytr_p = texts, cats, pris
    else:
        ytr_c, ytr_p = pick(cats, train_idx), pick(pris, train_idx)
    bundle = {
        "category_model": build_pipeline().fit(xtr, ytr_c),
        "priority_model": build_pipeline().fit(xtr, ytr_p),
        "sklearn_version": sklearn.__version__,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Category accuracy (test): {metrics['category']['accuracy']:.3f}  macro-F1 {metrics['category']['macro_f1']:.3f}")
    print(f"Priority accuracy (test): {metrics['priority']['accuracy']:.3f}  macro-F1 {metrics['priority']['macro_f1']:.3f}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true")
    ap.add_argument("--refit-full", action="store_true")
    a = ap.parse_args()
    main(a.regenerate, a.refit_full)
