
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def inspect_layer():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    query = """
    SELECT 
        RCPTNMBR,
        DATERECD,
        QTYRECVD,
        QTYSOLD,
        UNITCOST,
        ADJUNITCOST
    FROM IV10200
    WHERE RCPTNMBR = 'RV0000036070' AND ITEMNMBR = 'GRPASCORBIC'
    """
    
    df = pd.read_sql(query, conn)
    print(df.to_string())

if __name__ == "__main__":
    inspect_layer()
