import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

conn_str, server, db, auth = build_connection_string()
try:
    with pyodbc.connect(conn_str, autocommit=True) as conn:
        cursor = conn.cursor()
        
        # Query for items that look like the examples in the image
        query = """
        SELECT TOP 50 
            ITEMNMBR, 
            ITEMDESC, 
            USCATVLS_1 as Category,
            USCATVLS_2,
            STNDCOST,
            CURRCOST
        FROM IV00101 
        WHERE 
            ITEMNMBR LIKE 'CHEL%' OR 
            ITEMNMBR LIKE 'CITRIC%' OR 
            ITEMNMBR LIKE 'EDTA%' OR
            ITEMNMBR LIKE 'BORIC%' OR
            ITEMNMBR LIKE 'MAP%' OR
            ITEMNMBR LIKE 'MKP%'
        """
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        data = cursor.fetchall()
        df = pd.DataFrame.from_records(data, columns=columns)
        
        print("--- Potential Raw Materials ---")
        print(df.to_string())
        
        print("\n--- Unique Categories for these items ---")
        print(df['Category'].unique())

except Exception as e:
    print(f"Error: {e}")
