"""
Streamlit UI za predikciju broja iznajmljenih bicikala.

Pokretanje:
    streamlit run app/ui.py
"""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from config import TUNED_MODEL_PATH, MODEL_PATH

st.set_page_config(page_title="Bike Sharing Predikcija", page_icon="🚲", layout="centered")


@st.cache_resource
def load_model():
    path = TUNED_MODEL_PATH if TUNED_MODEL_PATH.exists() else MODEL_PATH
    return joblib.load(path), path.name


model, model_name = load_model()


def build_feature_row(inputs: dict) -> pd.DataFrame:
    base = {
        "yr": inputs["yr"],
        "mnth": inputs["mnth"],
        "hr": inputs["hr"],
        "holiday": inputs["holiday"],
        "weekday": inputs["weekday"],
        "workingday": inputs["workingday"],
        "temp": inputs["temp"],
        "atemp": inputs["atemp"],
        "hum": inputs["hum"],
        "windspeed": inputs["windspeed"],
    }
    for s in [1, 2, 3, 4]:
        base[f"season_{s}"] = int(inputs["season"] == s)
    for w in [1, 2, 3, 4]:
        base[f"weathersit_{w}"] = int(inputs["weathersit"] == w)

    if hasattr(model, "feature_names_in_"):
        row = {col: base.get(col, 0) for col in model.feature_names_in_}
    else:
        row = base

    return pd.DataFrame([row])


st.title("🚲 Bike Sharing — Predikcija iznajmljivanja")
st.caption(f"Model: `{model_name}`")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📅 Kalendarski podaci")
    yr = st.selectbox("Godina", options=[0, 1], format_func=lambda x: "2011" if x == 0 else "2012")
    season = st.selectbox("Sezona", options=[1, 2, 3, 4],
                          format_func=lambda x: {1: "Zima", 2: "Proleće", 3: "Leto", 4: "Jesen"}[x])
    mnth = st.slider("Mesec", 1, 12, 6)
    hr = st.slider("Sat dana", 0, 23, 17)
    weekday = st.selectbox("Dan u nedelji", options=list(range(7)),
                           format_func=lambda x: ["Ned", "Pon", "Uto", "Sre", "Čet", "Pet", "Sub"][x])
    holiday = st.selectbox("Praznik", options=[0, 1], format_func=lambda x: "Ne" if x == 0 else "Da")
    workingday = st.selectbox("Radni dan", options=[0, 1], format_func=lambda x: "Ne" if x == 0 else "Da")

with col2:
    st.subheader("🌤 Vremenski uslovi")
    weathersit = st.selectbox(
        "Vremenski uslovi",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "1 — Vedro / Malo oblacno",
            2: "2 — Magla / Oblacno",
            3: "3 — Lak sneg ili kisa",
            4: "4 — Jak sneg / Kisa / Oluja",
        }[x],
    )
    temp = st.slider("Temperatura (normalizovana)", 0.0, 1.0, 0.5, step=0.01,
                     help="0 = -8°C, 1 = 39°C (Capital Bikeshare skala)")
    atemp = st.slider("Osećajna temperatura (normalizovana)", 0.0, 1.0, 0.5, step=0.01)
    hum = st.slider("Vlažnost vazduha (normalizovana)", 0.0, 1.0, 0.5, step=0.01)
    windspeed = st.slider("Brzina vetra (normalizovana)", 0.0, 1.0, 0.2, step=0.01)

st.markdown("---")

if st.button("🔮 Predvidi broj iznajmljivanja", use_container_width=True, type="primary"):
    inputs = dict(
        yr=yr, season=season, mnth=mnth, hr=hr, weekday=weekday,
        holiday=holiday, workingday=workingday, weathersit=weathersit,
        temp=temp, atemp=atemp, hum=hum, windspeed=windspeed,
    )
    features = build_feature_row(inputs)
    prediction = max(0, int(round(model.predict(features)[0])))

    st.success(f"### Predvidjeni broj iznajmljenih bicikala: **{prediction}**")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Niski nivo", f"< 50")
    col_b.metric("Srednji nivo", "50 – 300")
    col_c.metric("Visoki nivo", "> 300")

    if prediction < 50:
        st.info("Niska potražnja — manje bicikala je potrebno rasporediti.")
    elif prediction < 300:
        st.warning("Srednja potražnja — standardna raspoređenost bicikala.")
    else:
        st.error("Visoka potražnja — preporučuje se maksimalna dostupnost bicikala.")

st.markdown("---")
st.caption("Bike Sharing SAUSAU projekat 2026 | Capital Bikeshare dataset (2011–2012)")
