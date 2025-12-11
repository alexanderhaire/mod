import pandas as pd
from app import _get_market_segment

# Mock data
data = {
    'ITEMNMBR': ['CHEL-100', 'PROD-200', 'EDTA-300', 'FG-400', 'CITRIC-500'],
    'ITEMDESC': ['Chelate', 'Product', 'EDTA Mix', 'Finished Good', 'Citric Acid'],
    'USCATVLS_1': ['CHE', 'FG', 'EDTA', 'FG', 'ACID'],
    'ITMCLSCD': ['RAWMATNT', 'FINISHED', 'RAWMATNTE', 'FINISHED', 'RAWMATT'],
    'CURRCOST': [10.0, 50.0, 15.0, 100.0, 5.0],
    'STNDCOST': [9.0, 45.0, 14.0, 90.0, 4.0],
    'PriceChange': [1.0, 5.0, 1.0, 10.0, 1.0],
    'PctChange': [11.1, 11.1, 7.1, 11.1, 25.0]
}

df_market = pd.DataFrame(data)

print("Original Data:")
print(df_market[['ITEMNMBR', 'USCATVLS_1', 'ITMCLSCD']])

# Simulate the logic added to app.py
df_ticker = df_market.copy()
df_ticker['Segment'] = df_ticker.apply(_get_market_segment, axis=1)
df_ticker = df_ticker[df_ticker['Segment'] == "Raw Material"]

print("\nFiltered Ticker Data (Should only be Raw Materials):")
print(df_ticker[['ITEMNMBR', 'USCATVLS_1', 'Segment']])

expected_items = {'CHEL-100', 'EDTA-300', 'CITRIC-500'}
actual_items = set(df_ticker['ITEMNMBR'])

if expected_items == actual_items:
    print("\nSUCCESS: Ticker logic correctly filtered for Raw Materials.")
else:
    print(f"\nFAILURE: Expected {expected_items}, got {actual_items}")
