import pandas as pd
import os

def inspect_external_prices():
    print("--- DIAGNOSTIC: External Pricing ---")
    
    files = [
        "Brenntag_Meeting_Prep_2026-01-28.xlsx",
        "10-01-25 Finished Goods and Rawmaterial Count.xls"
    ]
    
    for f in files:
        if os.path.exists(f):
            print(f"\nScanning {f}...")
            try:
                # read_excel might default to first sheet
                df = pd.read_excel(f)
                print(f"Columns: {list(df.columns)}")
                print(df.head(5).to_string())
            except Exception as e:
                print(f"Error reading {f}: {e}")
        else:
            print(f"File not found: {f}")

if __name__ == "__main__":
    inspect_external_prices()
