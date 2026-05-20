import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import mlflow

# Load
df = pd.read_csv("data/ds_salaries.csv")

# Select columns
features = ['work_year', 'experience_level', 'job_title', 
            'remote_ratio', 'company_location', 'company_size']
target = 'salary_in_usd'

df = df[features + [target]]

# Encode
categorical_cols = ['experience_level', 'job_title', 
                    'company_location', 'company_size']

import pickle

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save
os.makedirs("model", exist_ok=True)

with open("model/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("encoders saved")


# Split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


mlflow.start_run()
# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2  = r2_score(y_test, predictions)

print(f"MAE: ${mae:,.0f}")
print(f"R2:  {r2:.2f}")


mlflow.log_param("n_estimators", 100)
mlflow.log_metric("mae", mae)
mlflow.log_metric("r2", r2)
mlflow.end_run()



# Save
# os.makedirs("model", exist_ok=True)

# with open("model/model.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("model saved")