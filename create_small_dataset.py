# create_small_dataset.py
import pandas as pd

print("Loading original dataset...")
df = pd.read_csv('data/train.csv')

print(f"Original size: {len(df)} rows")

# Take a random sample - 50,000 rows (much faster to train)
df_small = df.sample(n=50000, random_state=42)

print(f"Sample size: {len(df_small)} rows")

# Save to data folder
df_small.to_csv('data/train.csv', index=False)

print("✅ Small dataset created at data/train.csv")