import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def check_layers():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # Check Receipt Layers (IV10200)
        query = """
        SELECT 
            DATERECD,
            QTYRECVD,
            QTYSOLD,
            QTYONHND,
            UNITCOST
        FROM IV10200
        WHERE ITEMNMBR = 'H2OCOLD'
          AND QTYONHND > 0
        ORDER BY DATERECD DESC
        """
        df = pd.read_sql(query, conn)
        print(df.to_string())
        
        total_val = (df['QTYONHND'] * df['UNITCOST']).sum()
        print(f"\nTotal Value in Layers: ${total_val:,.2f}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    check_layers()
