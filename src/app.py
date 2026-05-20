import os

from fastapi import FastAPI
from pydantic import BaseModel

import pickle

app = FastAPI()


os.makedirs("model", exist_ok=True)

with open("model/model.pkl","rb") as f:
    pickle.dump(model, f)

with open("model/encoders.pkl","rb") as f:
    pickle.dump(encoders, f)

class SalaryInput(BaseModel):
    work_year: int
    experience_level: str
    job_title: str
    remote_ratio:int
    company_location: str
    company_size: str


@app.get("/")
def root():
    return {"message": "PaySignal API is running"}


@app.post("/predict")
def predict(data: SalaryInput):
    
    # Encode strings to numbers
    exp = encoders['experience_level'].transform([data.experience_level])[0]
    job = encoders['job_title'].transform([data.job_title])[0]
    loc = encoders['company_location'].transform([data.company_location])[0]
    size = encoders['company_size'].transform([data.company_size])[0]

    # Build input for model
    import pandas as pd
    sample_df = pd.DataFrame([{
        'work_year': data.work_year,
        'experience_level': exp,
        'job_title': job,
        'remote_ratio': data.remote_ratio,
        'company_location': loc,
        'company_size': size
    }])

    # Predict
    predicted = model.predict(sample_df)[0]

    return {"predicted_salary_usd": round(predicted, 2)}