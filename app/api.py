"""
FastAPI - REST API za predikciju broja iznajmljenih bicikala.

Pokretanje:
    uvicorn app.api:app --reload

Primjer poziva:
    POST /predict
    {
      "season": 2, "yr": 1, "mnth": 6, "hr": 17,
      "holiday": 0, "weekday": 4, "workingday": 1,
      "weathersit": 1, "temp": 0.62, "atemp": 0.58,
      "hum": 0.4, "windspeed": 0.15
    }
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from typing import Literal
import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from config import TUNED_MODEL_PATH, MODEL_PATH

app = FastAPI(
    title="Bike Sharing Prediction API",
    description="Predvidjanje ukupnog broja iznajmljenih bicikala na osnovu vremenskih i kalendarskih karakteristika.",
    version="1.0.0",
)


def load_model():
    path = TUNED_MODEL_PATH if TUNED_MODEL_PATH.exists() else MODEL_PATH
    if not path.exists():
        raise RuntimeError(f"Model nije pronadjen. Pokrenite main.py prvo.")
    return joblib.load(path), path


model, model_path = load_model()


class BikeInput(BaseModel):
    season: int = Field(..., ge=1, le=4, description="Sezona: 1=Zima, 2=Prolece, 3=Leto, 4=Jesen")
    yr: int = Field(..., ge=0, le=1, description="Godina: 0=2011, 1=2012")
    mnth: int = Field(..., ge=1, le=12, description="Mesec (1-12)")
    hr: int = Field(..., ge=0, le=23, description="Sat dana (0-23)")
    holiday: int = Field(..., ge=0, le=1, description="Praznik: 0=Ne, 1=Da")
    weekday: int = Field(..., ge=0, le=6, description="Dan u nedelji (0=Nedjelja, 6=Subota)")
    workingday: int = Field(..., ge=0, le=1, description="Radni dan: 0=Ne, 1=Da")
    weathersit: int = Field(..., ge=1, le=4, description="Vremenski uslovi: 1=Vedro, 2=Magla, 3=Kisa/Sneg, 4=Ekstremno")
    temp: float = Field(..., ge=0.0, le=1.0, description="Temperatura (normalizovana 0-1)")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Osecajna temperatura (normalizovana 0-1)")
    hum: float = Field(..., ge=0.0, le=1.0, description="Vlaznost (normalizovana 0-1)")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Brzina vetra (normalizovana 0-1)")


def build_feature_row(inp: BikeInput) -> pd.DataFrame:
    base = {
        "yr": inp.yr, "mnth": inp.mnth, "hr": inp.hr,
        "holiday": inp.holiday, "weekday": inp.weekday,
        "workingday": inp.workingday,
        "temp": inp.temp, "atemp": inp.atemp,
        "hum": inp.hum, "windspeed": inp.windspeed,
    }

    for s in [1, 2, 3, 4]:
        base[f"season_{s}"] = int(inp.season == s)

    for w in [1, 2, 3, 4]:
        base[f"weathersit_{w}"] = int(inp.weathersit == w)

    expected_cols = [col for col in model.feature_names_in_] if hasattr(model, "feature_names_in_") else list(base.keys())
    row = {col: base.get(col, 0) for col in expected_cols}
    return pd.DataFrame([row])


@app.get("/")
def root():
    return {"message": "Bike Sharing API je aktivan", "model": str(model_path.name)}


@app.post("/predict")
def predict(inp: BikeInput):
    try:
        features = build_feature_row(inp)
        prediction = model.predict(features)[0]
        prediction = max(0, int(round(prediction)))
        return {
            "predicted_cnt": prediction,
            "opis": f"Predvidjeni broj iznajmljenih bicikala: {prediction}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
