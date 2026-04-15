import pyodbc
from db_pool import get_connection

def find_lignin():
    print("--- Finding Lignin Items ---")
    query = """
    SELECT TOP 50 ITEMNMBR, ITEMDESC 
    FROM IV00101 WITH (NOLOCK)
    WHERE ITEMDESC LIKE '%LIGNIN%' 
       OR ITEMDESC LIKE '%SULFONATE%'
       OR ITEMDESC LIKE '%SULPHONATE%'
    """
    
    with get_connection() as conn:
        cursor = conn.cursor()
        print("Executing query...")
        cursor.execute(query)
        rows = cursor.fetchall()
        
        print(f"Found {len(rows)} items:")
        for row in rows:
            print(f"- {row.ITEMNMBR.strip()}: {row.ITEMDESC.strip()}")

if __name__ == "__main__":
    find_lignin()
