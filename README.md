# Bike Sharing — Predikcija broja iznajmljenih bicikala

SAUSAU predmetni projekat 2026

Ovaj projekat implementira kompletan ML pipeline za predikciju ukupnog broja iznajmljenih bicikala (`cnt`) u Capital Bikeshare sistemu (Washington D.C.) na osnovu vremenskih, sezonskih i kalendarskih karakteristika. Dataset pokriva period 2011–2012 sa granulacijom po satu.

---

## Struktura projekta

```
projekat sausau/
├── config.py                    # Centralne putanje i konstante
├── main.py                      # Orkestratorski skript — pokreće ceo pipeline
├── requirements.txt             # Python zavisnosti
│
├── data/
│   ├── hour.csv                 # Originalni dataset
│   └── processed.csv            # Preprocesirani dataset (generiše se automatski)
│
├── models/
│   ├── best_model.joblib        # Baseline model (Random Forest)
│   └── tuned_model.joblib       # Finalni model (Gradient Boosting, tuned)
│
├── plots/                       # Grafici generisani tokom pipeline-a
│
├── src/
│   ├── preprocessing.py         # Čišćenje podataka, enkodiranje
│   ├── eda.py                   # Eksplorativna analiza, 9 grafika
│   ├── model_training.py        # Treniranje i poredjenje 5 modela
│   ├── hyperparameter_tuning.py # GridSearchCV optimizacija
│   └── feature_selection.py     # Analiza važnosti atributa
│
└── app/
    ├── api.py                   # FastAPI REST API
    └── ui.py                    # Streamlit web interfejs
```

---

## Rezultati

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 103.74 | 137.92 | 0.399 |
| Ridge Regression | 103.74 | 137.91 | 0.399 |
| Decision Tree | 35.78 | 58.72 | 0.891 |
| Random Forest | 24.89 | 42.04 | 0.944 |
| **Gradient Boosting (tuned)** | **25.52** | **40.80** | **0.947** |

Finalni model (Gradient Boosting sa podešenim hiperparametrima) postiže **R²=0.947**, što znači da objašnjava 94.7% varijanse u broju iznajmljivanja.

---

## Instalacija

### Preduslovi

- Python 3.10+

### Koraci

```bash
# 1. Klonirati repozitorijum
git clone <url-repozitorijuma>
cd "projekat sausau"

# 2. Instalirati zavisnosti
pip install -r requirements.txt
```

---

## Pokretanje

### Ceo ML pipeline

Pokreće sve faze redom: preprocesiranje → EDA → treniranje → tuning → feature selection.

```bash
python main.py
```

Za brže testiranje (preskočiti GridSearchCV koji traje ~5–10 minuta):

```bash
python main.py --skip-tuning
```

Faze se mogu pokretati i pojedinačno:

```bash
python src/preprocessing.py
python src/eda.py
python src/model_training.py
python src/hyperparameter_tuning.py
python src/feature_selection.py
```

---

### REST API (FastAPI)

```bash
python -m uvicorn app.api:app --reload
```

API je dostupan na `http://127.0.0.1:8000`

Interaktivna dokumentacija sa svim endpoint-ima: `http://127.0.0.1:8000/docs`

#### Primer zahteva

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "season": 3,
    "yr": 1,
    "mnth": 7,
    "hr": 17,
    "holiday": 0,
    "weekday": 4,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.72,
    "atemp": 0.67,
    "hum": 0.45,
    "windspeed": 0.12
  }'
```

#### Primer odgovora

```json
{
  "predicted_cnt": 387,
  "opis": "Predvidjeni broj iznajmljenih bicikala: 387"
}
```

---

### Web interfejs (Streamlit)

```bash
python -m streamlit run app/ui.py
```

Interfejs se otvara automatski na `http://localhost:8501`

---

## Opis atributa

| Atribut | Opis | Tip |
|---|---|---|
| `season` | Sezona (1=Zima, 2=Proleće, 3=Leto, 4=Jesen) | Kategorijski |
| `yr` | Godina (0=2011, 1=2012) | Binarni |
| `mnth` | Mesec (1–12) | Numerički |
| `hr` | Sat dana (0–23) | Numerički |
| `holiday` | Da li je praznik (0/1) | Binarni |
| `weekday` | Dan u nedelji (0=Ned, 6=Sub) | Numerički |
| `workingday` | Da li je radni dan (0/1) | Binarni |
| `weathersit` | Vremenski uslovi (1=Vedro … 4=Oluja) | Kategorijski |
| `temp` | Temperatura (normalizovana 0–1) | Numerički |
| `atemp` | Osećajna temperatura (normalizovana 0–1) | Numerički |
| `hum` | Vlažnost vazduha (normalizovana 0–1) | Numerički |
| `windspeed` | Brzina vetra (normalizovana 0–1) | Numerički |
| `cnt` | **Ciljna promenljiva** — ukupan broj iznajmljivanja | Numerički |

> **Napomena:** Atributi `instant`, `dteday`, `casual` i `registered` su uklonjeni tokom preprocesiranja (detalji u dokumentaciji).

---

## Ključni zaključci

- **Sat dana** (`hr`) je ubedljivo najvažniji prediktor sa ~60% feature importance — jutarnji (7–9h) i popodnevni (17–19h) špic dominiraju
- **Temperatura** je drugi najvažniji faktor (~9.5%) — više temperature povećavaju potražnju
- **Godina** (`yr`) beleži rast korišćenja od 2011 do 2012 (~8.8% importance)
- Top 10 atributa postiže gotovo isti R² (0.945) kao svih 18 atributa zajedno
- Linearni modeli su značajno lošiji (R²≈0.40) zbog nelinearnih veza u podacima
