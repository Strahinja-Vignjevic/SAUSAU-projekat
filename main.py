"""
Glavni orkestratorski skript — pokrecemo sve faze projekta redom.

Upotreba:
    python main.py              # ceo pipeline
    python main.py --skip-eda  # preskoci EDA grafike
"""

import argparse
import time


def run_step(label: str, fn, *args, **kwargs):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = fn(*args, **kwargs)
    print(f"\n  Zavrseno za {time.time() - t0:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description="Bike Sharing ML pipeline")
    parser.add_argument("--skip-eda", action="store_true", help="Preskoci EDA korak")
    parser.add_argument("--skip-tuning", action="store_true", help="Preskoci GridSearchCV (sporo)")
    args = parser.parse_args()

    from src.preprocessing import preprocess
    from src.model_training import train_models
    from src.feature_selection import run_feature_selection

    run_step("1 / 5  PREPROCESIRANJE PODATAKA", preprocess)

    if not args.skip_eda:
        from src.eda import run_eda
        run_step("2 / 5  EKSPLORATIVNA ANALIZA", run_eda)
    else:
        print("\n[PRESKOCEN] EDA korak")

    run_step("3 / 5  TRENIRANJE MODELA", train_models)

    if not args.skip_tuning:
        from src.hyperparameter_tuning import tune_models
        run_step("4 / 5  PODESAVANJE HIPERPARAMETARA", tune_models)
    else:
        print("\n[PRESKOCEN] Hyperparameter tuning korak")

    run_step("5 / 5  ANALIZA ZNACAJNOSTI ATRIBUTA", run_feature_selection)

    print("\n" + "=" * 60)
    print("  PIPELINE KOMPLETIRAN")
    print("  Modeli sacuvani u: models/")
    print("  Grafici sacuvani u: plots/")
    print()
    print("  Pokretanje API-ja:  uvicorn app.api:app --reload")
    print("  Pokretanje UI-ja:   streamlit run app/ui.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
