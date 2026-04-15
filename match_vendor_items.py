
import pandas as pd
import os

# Keywords to search for
keywords = [
    'potassium fulvate', 
    'sodium humate', 
    'humic', 
    'fulvic', 
    'gibberellic', 
    'abscisic', 
    'seaweed', 
    'compound npk', 
    'boron humate', 
    'amino acid', 
    'eddha',
    'soil amendment'
]

file_path = 'Iventory with quantities.xlsx'

if os.path.exists(file_path):
    print(f"--- Searching {file_path} ---")
    try:
        df = pd.read_excel(file_path)
        # Ensure ItemDescription is string
        df['ItemDescription'] = df['ItemDescription'].fillna('').astype(str)
        
        matches = []
        for index, row in df.iterrows():
            desc = row['ItemDescription'].lower()
            for k in keywords:
                if k in desc:
                    row['MatchedKeyword'] = k
                    matches.append(row)
                    break 
        
        print(f"Found {len(matches)} matches.")
        
        if matches:
            result_df = pd.DataFrame(matches)
            cols = ['ItemNumber', 'ItemDescription', 'QtyAvailable', 'MatchedKeyword']
            # print nicely
            print(result_df[cols].to_string(index=False))
            
            # Also searching Current GP inventory for more detailed check if needed
            # But let's stick to this one for now as it has Qty.
            
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"File not found: {file_path}")
