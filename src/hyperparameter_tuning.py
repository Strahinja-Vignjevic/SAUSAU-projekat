"""
Podesavanje hiperparametara za Random Forest i Gradient Boosting
koriscenjem GridSearchCV sa 5-fold cross-validacijom.
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DATA_PROCESSED, PLOTS_DIR, TUNED_MODEL_PATH, RANDOM_STATE, TEST_SIZE, TARGET


def load_processed() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_PROCESSED)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def tune_random_forest(X_train, y_train) -> tuple:
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", 0.5],
    }

    print("\n[Random Forest] GridSearchCV...")
    rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid_rf = GridSearchCV(
        rf, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )
    grid_rf.fit(X_train, y_train)
    print(f"  Najbolji parametri: {grid_rf.best_params_}")
    print(f"  Najbolji CV R²:     {grid_rf.best_score_:.4f}")
    return grid_rf.best_estimator_, grid_rf.best_params_, grid_rf.best_score_


def tune_gradient_boosting(X_train, y_train) -> tuple:
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
    }

    print("\n[Gradient Boosting] GridSearchCV...")
    gb = GradientBoostingRegressor(random_state=RANDOM_STATE)
    grid_gb = GridSearchCV(
        gb, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=1
    )
    grid_gb.fit(X_train, y_train)
    print(f"  Najbolji parametri: {grid_gb.best_params_}")
    print(f"  Najbolji CV R²:     {grid_gb.best_score_:.4f}")
    return grid_gb.best_estimator_, grid_gb.best_params_, grid_gb.best_score_


def evaluate(y_true, y_pred, label: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{label}] Test metrike:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def plot_cv_comparison(rf_score: float, gb_score: float) -> None:
    names = ["Random Forest (tuned)", "Gradient Boosting (tuned)"]
    scores = [rf_score, gb_score]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, scores, color=["steelblue", "coral"], edgecolor="white")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.02,
                 f"{score:.4f}", ha="center", va="top", color="white", fontweight="bold")
    plt.title("Cross-Validation R² - Podeseni modeli")
    plt.ylabel("CV R² (5-fold)")
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "12_tuned_model_comparison.png", dpi=150)
    plt.close()
    print("\nSacuvan: 12_tuned_model_comparison.png")


def tune_models() -> None:
    print("=" * 55)
    print("PODESAVANJE HIPERPARAMETARA")
    print("=" * 55)

    X, y = load_processed()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    rf_model, rf_params, rf_cv = tune_random_forest(X_train, y_train)
    gb_model, gb_params, gb_cv = tune_gradient_boosting(X_train, y_train)

    rf_metrics = evaluate(y_test, rf_model.predict(X_test), "Random Forest (tuned)")
    gb_metrics = evaluate(y_test, gb_model.predict(X_test), "Gradient Boosting (tuned)")

    plot_cv_comparison(rf_cv, gb_cv)

    if rf_metrics["R2"] >= gb_metrics["R2"]:
        best_model = rf_model
        best_label = "Random Forest"
    else:
        best_model = gb_model
        best_label = "Gradient Boosting"

    joblib.dump(best_model, TUNED_MODEL_PATH)
    print(f"\nNajbolji podeseni model: {best_label}")
    print(f"Sacuvan: {TUNED_MODEL_PATH}")


if __name__ == "__main__":
    tune_models()
