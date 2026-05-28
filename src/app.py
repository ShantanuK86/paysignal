from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import time
import json

# ─────────────────────────────────────────
# Structured JSON logger
# ─────────────────────────────────────────
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage()
        }
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        return json.dumps(log_data)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())

logger = logging.getLogger("paysignal")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ─────────────────────────────────────────
# App setup
# ─────────────────────────────────────────
app = FastAPI()

with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

logger.info("Model and encoders loaded successfully")

class SalaryInput(BaseModel):
    work_year: int
    experience_level: str
    job_title: str
    remote_ratio: int
    company_location: str
    company_size: str

# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "PaySignal API is running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "paysignal-salary-predictor",
        "encoders": list(encoders.keys()),
        "features": [
            "work_year",
            "experience_level", 
            "job_title",
            "remote_ratio",
            "company_location",
            "company_size"
        ]
    }
@app.post("/predict")
def predict(data: SalaryInput, request: Request):
    start_time = time.time()

    exp  = encoders['experience_level'].transform([data.experience_level])[0]
    job  = encoders['job_title'].transform([data.job_title])[0]
    loc  = encoders['company_location'].transform([data.company_location])[0]
    size = encoders['company_size'].transform([data.company_size])[0]

    sample_df = pd.DataFrame([{
        'work_year': data.work_year,
        'experience_level': exp,
        'job_title': job,
        'remote_ratio': data.remote_ratio,
        'company_location': loc,
        'company_size': size
    }])

    predicted = model.predict(sample_df)[0]
    duration  = round((time.time() - start_time) * 1000, 2)

    # Log every prediction
    log_record = logger.makeRecord(
        "paysignal", logging.INFO, "", 0,
        "prediction made", [], None
    )
    log_record.extra = {
        "experience_level": data.experience_level,
        "job_title": data.job_title,
        "company_location": data.company_location,
        "company_size": data.company_size,
        "remote_ratio": data.remote_ratio,
        "predicted_salary": round(predicted, 2),
        "duration_ms": duration
    }
    logger.handle(log_record)

    return {"predicted_salary_usd": round(predicted, 2)}