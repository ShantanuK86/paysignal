import pandas as pd

# Reference data — training distribution
reference = pd.read_csv("data/ds_salaries.csv")
ref_dist = reference['experience_level'].value_counts(normalize=True)
print("Training distribution:")
print(ref_dist)

# Simulated current data — what users are inputting now
# pretend 6 months passed, junior roles dried up
current_data = {
    'experience_level': ['SE'] * 70 + ['MI'] * 20 + ['EN'] * 5 + ['EX'] * 5
}
current = pd.DataFrame(current_data)
curr_dist = current['experience_level'].value_counts(normalize=True)
# print("\nCurrent user distribution:")
# print(curr_dist)


# Compare distributions
drift_threshold = 0.15  # if any category shifts more than 15%, flag it

print("\nDrift Report:")
drift_detected = False

for category in ['SE', 'MI', 'EN', 'EX']:
    ref   = ref_dist.get(category, 0)
    curr  = curr_dist.get(category, 0)
    delta = abs(curr - ref)

    status = "🚨 DRIFT" if delta > drift_threshold else "✅ OK"
    
    if delta > drift_threshold:
        drift_detected = True

    print(f"  {category}  ref={ref:.2f}  curr={curr:.2f}  delta={delta:.2f}  {status}")

print(f"\nRetrain needed: {drift_detected}")


import subprocess

if drift_detected:
    print("\n⚙️  Drift detected. Triggering retrain...")
    subprocess.run(["python", "src/train.py"])
    print("✅ Retrain complete. Fresh model saved.")
else:
    print("\n✅ No drift detected. Model is healthy.")