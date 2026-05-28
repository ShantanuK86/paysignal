import pickle
import pandas as pd

def test_model_loads():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    assert model is not None

def test_encoders_load():
    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    assert "experience_level" in encoders
    assert "job_title" in encoders
    assert "company_location" in encoders
    assert "company_size" in encoders

def test_prediction_returns_number():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    sample_df = pd.DataFrame([{
        'work_year': 2024,
        'experience_level': encoders['experience_level'].transform(['SE'])[0],
        'job_title': encoders['job_title'].transform(['Data Engineer'])[0],
        'remote_ratio': 100,
        'company_location': encoders['company_location'].transform(['US'])[0],
        'company_size': encoders['company_size'].transform(['L'])[0],
    }])

    predicted = model.predict(sample_df)[0]

    assert isinstance(predicted, float)
    assert predicted > 0
    assert predicted < 1000000