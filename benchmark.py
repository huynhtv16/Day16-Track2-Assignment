import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("creditcard.csv")
RESULT_PATH = Path("benchmark_result.json")


def timed(label, fn):
    start = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.4f}s")
    return value, elapsed


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH.resolve()}")

    df, load_time = timed("Load data", lambda: pd.read_csv(DATA_PATH))

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    _, training_time = timed(
        "Training",
        lambda: model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        ),
    )

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    one_row = X_test.iloc[[0]]
    start = time.perf_counter()
    model.predict_proba(one_row)
    one_row_latency_ms = (time.perf_counter() - start) * 1000

    batch = X_test.iloc[:1000]
    start = time.perf_counter()
    model.predict_proba(batch)
    batch_time = time.perf_counter() - start
    throughput_rows_per_sec = len(batch) / batch_time

    result = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "load_time_seconds": load_time,
        "training_time_seconds": training_time,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "inference_latency_1_row_ms": one_row_latency_ms,
        "inference_time_1000_rows_seconds": batch_time,
        "inference_throughput_rows_per_second": throughput_rows_per_sec,
    }

    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nBenchmark Results")
    print("=================")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"\nSaved: {RESULT_PATH.resolve()}")


if __name__ == "__main__":
    main()
