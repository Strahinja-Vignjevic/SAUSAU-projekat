"""
Eksplorativna analiza podataka (EDA) za Bike Sharing dataset.
Grafici se cuvaju u plots/ folderu.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from config import DATA_RAW, PLOTS_DIR

sns.set_theme(style="whitegrid", palette="muted")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW)


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["cnt"], bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Raspodela broja iznajmljivanja (cnt)")
    axes[0].set_xlabel("Broj iznajmljivanja")
    axes[0].set_ylabel("Frekvencija")

    axes[1].hist(np.log1p(df["cnt"]), bins=50, color="coral", edgecolor="white")
    axes[1].set_title("Log-raspodela cnt (log1p)")
    axes[1].set_xlabel("log(cnt + 1)")
    axes[1].set_ylabel("Frekvencija")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_target_distribution.png", dpi=150)
    plt.close()
    print("Sacuvan: 01_target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        annot_kws={"size": 7},
    )
    plt.title("Korelaciona matrica atributa", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_correlation_heatmap.png", dpi=150)
    plt.close()
    print("Sacuvan: 02_correlation_heatmap.png")

    print("\nNajjace korelacije sa cnt:")
    cnt_corr = corr["cnt"].drop("cnt").abs().sort_values(ascending=False)
    print(cnt_corr.head(10).to_string())


def plot_cnt_by_season(df: pd.DataFrame) -> None:
    season_labels = {1: "Zima", 2: "Prolece", 3: "Leto", 4: "Jesen"}
    df = df.copy()
    df["season_label"] = df["season"].map(season_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    avg = df.groupby("season_label")["cnt"].mean().reindex(["Zima", "Prolece", "Leto", "Jesen"])
    axes[0].bar(avg.index, avg.values, color=["#4e79a7", "#59a14f", "#f28e2b", "#e15759"])
    axes[0].set_title("Prosecno iznajmljivanje po sezoni")
    axes[0].set_xlabel("Sezona")
    axes[0].set_ylabel("Prosecni cnt")

    df.boxplot(column="cnt", by="season_label", ax=axes[1],
               positions=[1, 2, 3, 4], patch_artist=True)
    axes[1].set_title("Raspodela cnt po sezoni")
    axes[1].set_xlabel("Sezona")
    axes[1].set_ylabel("cnt")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "03_cnt_by_season.png", dpi=150)
    plt.close()
    print("Sacuvan: 03_cnt_by_season.png")


def plot_cnt_by_hour(df: pd.DataFrame) -> None:
    avg_by_hour = df.groupby(["hr", "workingday"])["cnt"].mean().reset_index()
    workday = avg_by_hour[avg_by_hour["workingday"] == 1]
    weekend = avg_by_hour[avg_by_hour["workingday"] == 0]

    plt.figure(figsize=(12, 5))
    plt.plot(workday["hr"], workday["cnt"], marker="o", label="Radni dan", color="steelblue")
    plt.plot(weekend["hr"], weekend["cnt"], marker="s", label="Vikend/Praznik", color="coral")
    plt.title("Prosecno iznajmljivanje po satu dana")
    plt.xlabel("Sat (0-23)")
    plt.ylabel("Prosecni cnt")
    plt.xticks(range(0, 24))
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_cnt_by_hour.png", dpi=150)
    plt.close()
    print("Sacuvan: 04_cnt_by_hour.png")


def plot_cnt_by_month(df: pd.DataFrame) -> None:
    avg = df.groupby("mnth")["cnt"].mean()

    plt.figure(figsize=(10, 4))
    plt.bar(avg.index, avg.values, color="teal", edgecolor="white")
    plt.title("Prosecno iznajmljivanje po mesecu")
    plt.xlabel("Mesec")
    plt.ylabel("Prosecni cnt")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun",
                               "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"])
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_cnt_by_month.png", dpi=150)
    plt.close()
    print("Sacuvan: 05_cnt_by_month.png")


def plot_cnt_by_weather(df: pd.DataFrame) -> None:
    weather_labels = {
        1: "Vedro",
        2: "Magla/Oblacno",
        3: "Lak sneg/Kisa",
        4: "Jak sneg/Kisa",
    }
    df = df.copy()
    df["weather_label"] = df["weathersit"].map(weather_labels)

    plt.figure(figsize=(9, 5))
    order = ["Vedro", "Magla/Oblacno", "Lak sneg/Kisa", "Jak sneg/Kisa"]
    avg = df.groupby("weather_label")["cnt"].mean().reindex(order)
    colors = ["#59a14f", "#f28e2b", "#4e79a7", "#e15759"]
    plt.bar(avg.index, avg.values, color=colors, edgecolor="white")
    plt.title("Prosecno iznajmljivanje po vremenskim uslovima")
    plt.xlabel("Vremenski uslovi")
    plt.ylabel("Prosecni cnt")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_cnt_by_weather.png", dpi=150)
    plt.close()
    print("Sacuvan: 06_cnt_by_weather.png")


def plot_cnt_vs_temperature(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(df["temp"], df["cnt"], alpha=0.2, s=5, color="steelblue")
    axes[0].set_title("cnt vs Temperatura")
    axes[0].set_xlabel("temp (normalizovana)")
    axes[0].set_ylabel("cnt")

    axes[1].scatter(df["hum"], df["cnt"], alpha=0.2, s=5, color="coral")
    axes[1].set_title("cnt vs Vlaznost")
    axes[1].set_xlabel("hum (normalizovana)")
    axes[1].set_ylabel("cnt")

    axes[2].scatter(df["windspeed"], df["cnt"], alpha=0.2, s=5, color="green")
    axes[2].set_title("cnt vs Brzina vetra")
    axes[2].set_xlabel("windspeed (normalizovana)")
    axes[2].set_ylabel("cnt")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_cnt_vs_weather_numeric.png", dpi=150)
    plt.close()
    print("Sacuvan: 07_cnt_vs_weather_numeric.png")


def plot_cnt_by_weekday_and_holiday(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    weekday_labels = ["Ned", "Pon", "Uto", "Sre", "Cet", "Pet", "Sub"]
    avg_weekday = df.groupby("weekday")["cnt"].mean()
    axes[0].bar(weekday_labels, avg_weekday.values, color="mediumpurple", edgecolor="white")
    axes[0].set_title("Prosecno iznajmljivanje po danu u nedelji")
    axes[0].set_xlabel("Dan")
    axes[0].set_ylabel("Prosecni cnt")

    avg_holiday = df.groupby("holiday")["cnt"].mean()
    axes[1].bar(["Radni dan", "Praznik"], avg_holiday.values, color=["steelblue", "coral"], edgecolor="white")
    axes[1].set_title("Prosecno iznajmljivanje: Radni dan vs Praznik")
    axes[1].set_xlabel("Tip dana")
    axes[1].set_ylabel("Prosecni cnt")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_cnt_by_weekday_holiday.png", dpi=150)
    plt.close()
    print("Sacuvan: 08_cnt_by_weekday_holiday.png")


def plot_yearly_trend(df: pd.DataFrame) -> None:
    avg = df.groupby(["yr", "mnth"])["cnt"].mean().reset_index()
    yr0 = avg[avg["yr"] == 0]
    yr1 = avg[avg["yr"] == 1]

    plt.figure(figsize=(11, 4))
    plt.plot(yr0["mnth"], yr0["cnt"], marker="o", label="2011", color="steelblue")
    plt.plot(yr1["mnth"], yr1["cnt"], marker="s", label="2012", color="coral")
    plt.title("Mesecni trend iznajmljivanja po godini")
    plt.xlabel("Mesec")
    plt.ylabel("Prosecni cnt")
    plt.xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun",
                               "Jul", "Avg", "Sep", "Okt", "Nov", "Dec"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "09_yearly_trend.png", dpi=150)
    plt.close()
    print("Sacuvan: 09_yearly_trend.png")


def run_eda() -> None:
    print("=" * 50)
    print("EKSPLORATIVNA ANALIZA PODATAKA")
    print("=" * 50)

    df = load_raw()
    print(f"\nUcitano {df.shape[0]} zapisa, {df.shape[1]} atributa")

    plot_target_distribution(df)
    plot_correlation_heatmap(df)
    plot_cnt_by_season(df)
    plot_cnt_by_hour(df)
    plot_cnt_by_month(df)
    plot_cnt_by_weather(df)
    plot_cnt_vs_temperature(df)
    plot_cnt_by_weekday_and_holiday(df)
    plot_yearly_trend(df)

    print(f"\nSvi grafici sacuvani u: {PLOTS_DIR}")


if __name__ == "__main__":
    run_eda()
