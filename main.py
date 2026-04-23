"""Drift Happens: a tiny, presentation-ready ML demo.

Demo 1: Simple vs Complex model on the same classification task.
Demo 2: Data drift impact on model performance.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_SEED = 42


def create_base_dataset(seed: int = RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Create a stable binary classification dataset."""
    X, y = make_classification(
        n_samples=2000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        class_sep=1.1,
        flip_y=0.01,
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
    """Train a more complex model."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X: np.ndarray, y: np.ndarray) -> float:
    """Return accuracy score."""
    preds = model.predict(X)
    return accuracy_score(y, preds)


def simulate_drift(X: np.ndarray, seed: int = RANDOM_SEED) -> np.ndarray:
    """Create a drifted copy of X by shifting means and adding controlled noise."""
    rng = np.random.default_rng(seed)
    X_drift = X.copy()

    # Mean shift on selected features.
    X_drift[:, 0] += 0.7
    X_drift[:, 3] -= 0.5

    # Slight scale change on one feature.
    X_drift[:, 1] *= 1.25

    # Add small Gaussian noise to all features.
    noise = rng.normal(loc=0.0, scale=0.35, size=X_drift.shape)
    X_drift += noise

    return X_drift


def run_demo_simple_vs_complex() -> None:
    """Demo 1: compare simple and complex models."""
    print("\n" + "=" * 70)
    print("DEMO 1: Simple vs Complex Model")
    print("=" * 70)
    print("Creating dataset and splitting into train/test...")

    X, y = create_base_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("Training Logistic Regression (simple baseline)...")
    simple_model = train_simple_model(X_train, y_train)

    print("Training Random Forest (more complex model)...")
    complex_model = train_complex_model(X_train, y_train)

    simple_acc = evaluate_model(simple_model, X_test, y_test)
    complex_acc = evaluate_model(complex_model, X_test, y_test)

    print("\nResults on the same test data:")
    print(f"Logistic Regression Accuracy: {simple_acc:.3f}")
    print(f"Random Forest Accuracy:      {complex_acc:.3f}")

    gap = complex_acc - simple_acc
    print(f"Accuracy Gap (Complex - Simple): {gap:+.3f}")

    if abs(gap) <= 0.03:
        print("Takeaway: The simple model performs similarly to the complex one.")
    else:
        print("Takeaway: The complex model is better here, but not always worth complexity.")


def run_demo_data_drift() -> None:
    """Demo 2: show how data drift hurts production performance."""
    print("\n" + "=" * 70)
    print("DEMO 2: Data Drift in Production")
    print("=" * 70)

    X, y = create_base_dataset()
    X_train, X_test_a, y_train, y_test_a = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print("Model trained on old data (Dataset A)...")
    model = train_simple_model(X_train, y_train)

    print("Evaluating on familiar data distribution (Dataset A)...")
    acc_a = evaluate_model(model, X_test_a, y_test_a)

    print("Creating new data with drift (Dataset B)...")
    X_test_b = simulate_drift(X_test_a)

    print("Evaluating the same model on drifted data (Dataset B)...")
    acc_b = evaluate_model(model, X_test_b, y_test_a)

    drop = acc_a - acc_b

    print("\nDrift Results:")
    print(f"Accuracy on Dataset A (old distribution): {acc_a:.3f}")
    print(f"Accuracy on Dataset B (with drift):       {acc_b:.3f}")
    print(f"Performance Drop After Drift:             {drop:.3f}")

    print("Explanation:")
    print("- Model trained on old data")
    print("- New data has drift")
    print("- Performance dropped")


def main() -> None:
    print("\nDrift Happens: Rules of ML Demo")
    print("This script demonstrates why simple baselines and drift monitoring matter.")

    run_demo_simple_vs_complex()
    run_demo_data_drift()

    print("\nDone. Deterministic run complete (fixed random seed).")


if __name__ == "__main__":
    main()
