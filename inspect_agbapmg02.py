import pyodbc
from secrets_loader import build_connection_string

def inspect_agbapmg02():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        items = ['AGBAPMG02', 'NO3CA']
        placeholders = ','.join('?' for _ in items)
        
        query = f"""
        SELECT ITEMNMBR, ITEMDESC, ITMCLSCD, USCATVLS_1, ITEMTYPE, USCATVLS_2
        FROM IV00101 
        WHERE ITEMNMBR IN ({placeholders})
        """
        
        cursor.execute(query, items)
        rows = cursor.fetchall()
        
        print(f"{'ITEM':<15} {'CLASS':<15} {'CATEGORY':<15} {'TYPE':<5} {'CAT2'}")
        print("-" * 70)
        for r in rows:
            print(f"{r.ITEMNMBR:<15} {r.ITMCLSCD:<15} {r.USCATVLS_1:<15} {r.ITEMTYPE:<5} {r.USCATVLS_2}")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    inspect_agbapmg02()
