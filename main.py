"""Churn Prediction demo aligned with Rules of ML.

Demo 1: Simple vs Complex model on churn prediction.
Demo 2: Production drift impact on churn model performance.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def create_churn_training_data(seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic customer data for a churn prediction use case."""
    X, y = make_classification(
        n_samples=2200,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=1.9,
        flip_y=0.02,
        random_state=seed,
    )
    return X, y


def train_simple_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a simple baseline model."""
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def train_complex_model(
    X_train: np.ndarray, y_train: np.ndarray
) -> RandomForestClassifier:
    """Train a constrained complex model to avoid unnecessary advantage."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> float:
    """Return accuracy score."""
    preds = model.predict(X)
    return accuracy_score(y, preds)


def simulate_production_drift(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = RANDOM_SEED,
    label_flip_rate: float = 0.07,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate realistic production drift in customer behavior and serving."""
    rng = np.random.default_rng(seed)
    X_drift = X.copy()

    # Customer behavior shift over time (freshness issue).
    X_drift[:, 1] += 1.0
    X_drift[:, 2] *= 0.7
    X_drift[:, 4] += 0.8
    X_drift[:, 7] *= 0.75

    # Training/serving skew: one feature gets a new upstream scaling.
    X_drift[:, 3] *= 1.25

    noise = rng.normal(loc=0.0, scale=0.35, size=X_drift.shape)
    X_drift += noise

    y_drift = y.copy()
    flip_mask = rng.random(y_drift.shape[0]) < label_flip_rate
    y_drift[flip_mask] = 1 - y_drift[flip_mask]

    return X_drift, y_drift


def plot_comparison_bars(
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
) -> None:
    """Render a clean accuracy bar chart with value labels."""
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, values, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.show()


def run_demo_simple_vs_complex() -> None:
    """Demo 1: compare simple and complex models for churn prediction."""
    X, y = create_churn_training_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    simple_model = train_simple_model(X_train, y_train)
    complex_model = train_complex_model(X_train, y_train)

    simple_acc = evaluate_model(simple_model, X_test, y_test)
    complex_acc = evaluate_model(complex_model, X_test, y_test)

    gap = complex_acc - simple_acc
    print(f"Demo 1 - Logistic Regression accuracy: {simple_acc:.3f}")
    print(f"Demo 1 - Random Forest accuracy:      {complex_acc:.3f}")
    print(f"Demo 1 - Gap (complex - simple):      {gap:+.3f}")

    plot_comparison_bars(
        title="Simple vs Complex Model (Churn Prediction)",
        labels=["Logistic", "RandomForest"],
        values=[simple_acc, complex_acc],
        colors=["#1f77b4", "#ff7f0e"],
    )
    print(
        "Simple model performs similarly and is easier to debug -> aligns with Rule #4 and #14"
    )


def run_demo_data_drift() -> None:
    """Demo 2: show production drift impact in churn prediction."""
    X, y = create_churn_training_data()

    X_train, X_test_a, y_train, y_test_a = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    model = train_simple_model(X_train, y_train)
    acc_a = evaluate_model(model, X_test_a, y_test_a)

    X_test_b, y_test_b = simulate_production_drift(X_test_a, y_test_a)
    acc_b = evaluate_model(model, X_test_b, y_test_b)

    drop = acc_a - acc_b

    print(f"Demo 2 - Accuracy before drift: {acc_a:.3f}")
    print(f"Demo 2 - Accuracy after drift:  {acc_b:.3f}")
    print(f"Demo 2 - Performance drop:      {drop:.3f}")

    plot_comparison_bars(
        title="Model Degradation After Data Drift",
        labels=["Dataset A", "Dataset B (drift)"],
        values=[acc_a, acc_b],
        colors=["#2ca02c", "#d62728"],
    )
    print("Model performance drops due to data drift -> aligns with Rule #8 and #37")


def main() -> None:
    run_demo_simple_vs_complex()
    run_demo_data_drift()


if __name__ == "__main__":
    main()
