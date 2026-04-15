
import pandas as pd
import os

files = ['Current GP inventory.xlsx', 'Iventory with quantities.xlsx']

for f in files:
    if os.path.exists(f):
        print(f"--- Analyzing {f} ---")
        try:
            df = pd.read_excel(f)
            print("Columns:", df.columns.tolist())
            print("First 5 rows:")
            print(df.head())
        except Exception as e:
            print(f"Error reading {f}: {e}")
