"""
Preprocessing modula za Bike Sharing dataset.

Odluke o atributima:
- instant: uklonjen - sekvencionalni ID bez prediktivne vrednosti
- dteday: uklonjen - redundantan, informacija je vec sadrzana u yr, mnth, hr
- casual, registered: uklonjeni - direktno sumiraju na cnt (data leakage)
- season, weathersit: one-hot enkodiranje (nominalne kategorije)
- yr, mnth, hr, holiday, weekday, workingday: zadrzani kao numericki
- temp, atemp, hum, windspeed: vec normalizovani [0,1], zadrzani
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("PROVERA KVALITETA PODATAKA")
    print("=" * 50)

    print(f"\nBroj redova: {df.shape[0]}, Broj kolona: {df.shape[1]}")

    missing = df.isnull().sum()
    print(f"\nNedostajuce vrednosti:\n{missing[missing > 0] if missing.any() else 'Nema nedostajucih vrednosti'}")

    duplicates = df.duplicated().sum()
    print(f"\nBroj duplikata: {duplicates}")

    print(f"\nTip podataka:\n{df.dtypes}")

    print(f"\nStatisticki opis numerickih atributa:\n{df.describe()}")

    print(f"\nRaspodela ciljne promenljive (cnt):")
    print(f"  Min:  {df['cnt'].min()}")
    print(f"  Max:  {df['cnt'].max()}")
    print(f"  Mean: {df['cnt'].mean():.2f}")
    print(f"  Std:  {df['cnt'].std():.2f}")
    print(f"  Skewness: {df['cnt'].skew():.4f}")

    print(f"\nAnomaliije - cnt=0 (nema iznajmljivanja): {(df['cnt'] == 0).sum()} zapisa")
    print(f"Provjera: casual+registered == cnt: {(df['casual'] + df['registered'] == df['cnt']).all()}")


def drop_leaky_and_irrelevant(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ["instant", "dteday", "casual", "registered"]
    df = df.drop(columns=cols_to_drop)
    print(f"\nUklonjene kolone: {cols_to_drop}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    ohe_cols = ["season", "weathersit"]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=False, dtype=int)
    print(f"\nOne-hot enkodiranje primenjeno na: {ohe_cols}")
    return df


def preprocess(save: bool = True) -> pd.DataFrame:
    print("\nUCITAVANJE PODATAKA...")
    df = load_data()

    check_data_quality(df)

    df = drop_leaky_and_irrelevant(df)
    df = encode_categoricals(df)

    if save:
        df.to_csv(DATA_PROCESSED, index=False)
        print(f"\nPreprocesirani podaci sacuvani u: {DATA_PROCESSED}")

    print(f"\nFinalne kolone ({len(df.columns)}): {list(df.columns)}")
    return df


if __name__ == "__main__":
    preprocess()
