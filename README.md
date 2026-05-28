# 💰 PaySignal — AI-Era Salary Intelligence

Know your worth in the post-AI job market.

## Try it live
[Link to your Streamlit app]

## What it does
Predicts salary range based on role, experience, 
location, and company size — with AI-era market insights.

## MLOps Stack
| Component | Tool |
|---|---|
| Model training | scikit-learn RandomForest |
| Experiment tracking | MLflow |
| API serving | FastAPI |
| Automated retraining | GitHub Actions |
| Drift monitoring | Custom + Evidently |
| Public UI | Streamlit |

## How the pipeline works
1. Model trains on 600+ real salary data points
2. Every push triggers retraining via GitHub Actions
3. Every Monday drift monitor checks for distribution shift
4. If drift detected → automatic retrain → fresh model
5. Users always get predictions from current model