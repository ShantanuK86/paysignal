import pickle
import pandas as pd

# Load
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

print("model loaded")
print("encoders loaded")

# Sample input
sample = {
    'work_year': 2024,
    'experience_level': 'SE',
    'job_title': 'Data Engineer',
    'remote_ratio': 100,
    'company_location': 'US',
    'company_size': 'L'
}

# Encode
sample['experience_level'] = encoders['experience_level'].transform(['SE'])[0]
sample['job_title'] = encoders['job_title'].transform(['Data Engineer'])[0]
sample['company_location'] = encoders['company_location'].transform(['US'])[0]
sample['company_size'] = encoders['company_size'].transform(['L'])[0]

# Predict
sample_df = pd.DataFrame([sample])
predicted = model.predict(sample_df)[0]

print(f"Predicted salary: ${predicted:,.0f}")