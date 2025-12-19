
import pyodbc
import pandas as pd
from secrets_loader import build_connection_string

def check_item_setup():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    
    query = """
    SELECT 
        ITEMNMBR, 
        ITEMDESC, 
        STNDCOST, 
        CURRCOST, 
        VCTNMTHD, 
        UOMSCHDL
    FROM IV00101
    WHERE ITEMNMBR = 'GRPASCORBIC'
    """
    
    df = pd.read_sql(query, conn)
    print(df.to_string())

if __name__ == "__main__":
    check_item_setup()
