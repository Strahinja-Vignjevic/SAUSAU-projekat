"""
Treniranje i poredjenje vise regresionih modela na Bike Sharing datasetu.

Modeli:
  - Linear Regression
  - Ridge Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor

Metrike: MAE, RMSE, R²
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_PROCESSED, PLOTS_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE, TARGET


def load_processed() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PROCESSED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_compare(X_train, X_test, y_train, y_test) -> tuple[dict, dict]:
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE),
    }

    results = {}
    trained_models = {}

    print("\n{:<22} {:>10} {:>10} {:>10}".format("Model", "MAE", "RMSE", "R²"))
    print("-" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred)
        results[name] = metrics
        trained_models[name] = model
        print("{:<22} {:>10.2f} {:>10.2f} {:>10.4f}".format(
            name, metrics["MAE"], metrics["RMSE"], metrics["R2"]
        ))

    return results, trained_models


def plot_model_comparison(results: dict) -> None:
    names = list(results.keys())
    maes = [results[n]["MAE"] for n in names]
    rmses = [results[n]["RMSE"] for n in names]
    r2s = [results[n]["R2"] for n in names]

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(x, maes, color="steelblue", edgecolor="white")
    axes[0].set_title("MAE (manje = bolje)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("MAE")

    axes[1].bar(x, rmses, color="coral", edgecolor="white")
    axes[1].set_title("RMSE (manje = bolje)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("RMSE")

    axes[2].bar(x, r2s, color="green", edgecolor="white")
    axes[2].set_title("R² (vece = bolje)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_ylabel("R²")
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "10_model_comparison.png", dpi=150)
    plt.close()
    print("\nSacuvan: 10_model_comparison.png")


def plot_predictions_vs_actual(y_test, y_pred, model_name: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color="steelblue")
    lim = max(y_test.max(), y_pred.max())
    plt.plot([0, lim], [0, lim], "r--", label="Idealna predikcija")
    plt.title(f"Stvarne vs Predvidjene vrednosti - {model_name}")
    plt.xlabel("Stvarni cnt")
    plt.ylabel("Predvidjeni cnt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "11_predictions_vs_actual.png", dpi=150)
    plt.close()
    print("Sacuvan: 11_predictions_vs_actual.png")


def train_models() -> None:
    print("=" * 55)
    print("TRENIRANJE I POREDJENJE MODELA")
    print("=" * 55)

    X, y = load_processed()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nTrening skup: {X_train.shape[0]} uzoraka")
    print(f"Test skup:    {X_test.shape[0]} uzoraka")

    results, trained_models = train_and_compare(X_train, X_test, y_train, y_test)
    plot_model_comparison(results)

    best_name = max(results, key=lambda n: results[n]["R2"])
    best_model = trained_models[best_name]
    print(f"\nNajbolji model: {best_name}  (R²={results[best_name]['R2']:.4f})")

    y_pred_best = best_model.predict(X_test)
    plot_predictions_vs_actual(y_test, y_pred_best, best_name)

    joblib.dump(best_model, MODEL_PATH)
    print(f"Sacuvan best model: {MODEL_PATH}")

    return results, best_name


if __name__ == "__main__":
    train_models()
