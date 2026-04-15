import pandas as pd

try:
    df = pd.read_excel("c:/Users/alexh/Downloads/mod/10-01-25 Finished Goods and Rawmaterial Count.xls")
    print("Columns:", df.columns.tolist())
    print("First 5 rows:")
    print(df.head().to_string())
except Exception as e:
    print(e)
