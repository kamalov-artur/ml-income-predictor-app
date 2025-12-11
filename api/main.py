import joblib
import pandas as pd
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.pkl")

df_dirty = pd.read_csv("../data/adult.csv")
df_dirty = df_dirty.rename(columns={"marital-status": "marital_status"})
df = df_dirty.replace('?', np.nan).dropna()


class Person(BaseModel):
    age: int
    fnlwgt: int
    education_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str


@app.post("/predict")
def predict(data: Person):
    df_in = pd.DataFrame([data.model_dump()])
    pred = model.predict(df_in)[0]
    return {"income_greater_than_50k": bool(pred)}


@app.get("/person_categories")
def get_categories():
    return {
        "workclass": sorted(df["workclass"].dropna().unique().tolist()),
        "education": sorted(df["education"].dropna().unique().tolist()),
        "marital_status": sorted(df["marital_status"].dropna().unique().tolist()),
        "occupation": sorted(df["occupation"].dropna().unique().tolist()),
        "relationship": sorted(df["relationship"].dropna().unique().tolist()),
        "race": sorted(df["race"].dropna().unique().tolist()),
        "sex": sorted(df["sex"].dropna().unique().tolist()),
    }
