"""
Analiza znacajnosti atributa i poredjenje modela:
- Svi atributi vs. samo N najznacajnijih atributa
- Feature importance iz Random Forest modela
- Permutation importance kao dodatna validacija
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_PROCESSED, PLOTS_DIR, TUNED_MODEL_PATH, RANDOM_STATE, TEST_SIZE, TARGET

TOP_N = 10


def load_processed() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PROCESSED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def get_feature_importances(model, feature_names: list) -> pd.Series:
    importances = model.feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(importances: pd.Series, title: str, filename: str) -> None:
    plt.figure(figsize=(10, 6))
    importances.head(20).sort_values().plot(kind="barh", color="steelblue", edgecolor="white")
    plt.title(title)
    plt.xlabel("Znacajnost atributa")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"Sacuvan: {filename}")


def plot_permutation_importance(perm_imp, feature_names: list, filename: str) -> None:
    imp_mean = perm_imp.importances_mean
    sorted_idx = imp_mean.argsort()[-20:]

    plt.figure(figsize=(10, 6))
    plt.barh(
        [feature_names[i] for i in sorted_idx],
        imp_mean[sorted_idx],
        color="coral",
        edgecolor="white",
    )
    plt.title("Permutation Importance (top 20)")
    plt.xlabel("Srednja promena R² pri permutaciji")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()
    print(f"Sacuvan: {filename}")


def plot_all_vs_top(all_metrics: dict, top_metrics: dict) -> None:
    labels = ["MAE", "RMSE", "R²"]
    all_vals = [all_metrics["MAE"], all_metrics["RMSE"], all_metrics["R2"]]
    top_vals = [top_metrics["MAE"], top_metrics["RMSE"], top_metrics["R2"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, all_vals, width, label="Svi atributi", color="steelblue", edgecolor="white")
    ax.bar(x + width / 2, top_vals, width, label=f"Top {TOP_N} atributa", color="coral", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(f"Svi atributi vs. Top {TOP_N} najznacajnijih atributa")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "15_all_vs_top_features.png", dpi=150)
    plt.close()
    print("Sacuvan: 15_all_vs_top_features.png")


def run_feature_selection() -> None:
    print("=" * 55)
    print("ANALIZA ZNACAJNOSTI ATRIBUTA")
    print("=" * 55)

    X, y = load_processed()
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Pokusaj ucitavanja podesenog modela, inace treniraj novi RF
    try:
        model = joblib.load(TUNED_MODEL_PATH)
        print(f"\nUcitan podeseni model: {TUNED_MODEL_PATH}")
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Model nema feature_importances_")
    except Exception:
        print("\nTreniram Random Forest za analizu znacajnosti...")
        model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_train, y_train)

    # Feature importance iz modela
    importances = get_feature_importances(model, feature_names)
    print("\nTop 10 najznacajnijih atributa (model importance):")
    print(importances.head(TOP_N).to_string())

    plot_feature_importance(importances, "Feature Importance - Random Forest", "13_feature_importance.png")

    # Permutation importance
    print("\nRacunanje permutation importance (moze trajati par sekundi)...")
    perm_imp = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    plot_permutation_importance(perm_imp, feature_names, "14_permutation_importance.png")

    # Poredjenje: svi atributi vs top N
    top_features = importances.head(TOP_N).index.tolist()
    print(f"\nTop {TOP_N} atributa: {top_features}")

    all_metrics = evaluate(y_test, model.predict(X_test))

    rf_top = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_top.fit(X_train[top_features], y_train)
    top_metrics = evaluate(y_test, rf_top.predict(X_test[top_features]))

    print(f"\n{'Metrika':<8} {'Svi atributi':>16} {'Top ' + str(TOP_N) + ' atributa':>16}")
    print("-" * 42)
    for metric in ["MAE", "RMSE", "R2"]:
        print(f"{metric:<8} {all_metrics[metric]:>16.4f} {top_metrics[metric]:>16.4f}")

    plot_all_vs_top(all_metrics, top_metrics)

    diff = top_metrics["R2"] - all_metrics["R2"]
    print(f"\nRazlika R² (top - svi): {diff:+.4f}")
    if abs(diff) < 0.01:
        print(f"Zakljucak: Top {TOP_N} atributa postize gotovo iste performanse uz manji model.")
    elif diff < 0:
        print(f"Zakljucak: Koriscenje svih atributa daje bolji R² za {abs(diff):.4f}.")


if __name__ == "__main__":
    run_feature_selection()
