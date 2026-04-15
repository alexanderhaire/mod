import pyodbc
from secrets_loader import build_connection_string

def check_good_mo():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        # IV Doc that is definitely Posted (from previous step)
        iv_doc = '000148023'
        
        print(f"--- Checking MOP10213 for IVDoc {iv_doc} ---")
        query = "SELECT * FROM MOP10213 WHERE IVDOCNBR = ?"
        cursor.execute(query, iv_doc)
        rows = cursor.fetchall()
        if rows:
            columns = [column[0] for column in cursor.description]
            print(columns)
            for r in rows:
                print(list(r))
                mo_num = r.MANUFACTUREORDER_I
                print(f"  Found MO: {mo_num}")
                
                # Check status of this MO
                print(f"  Checking Status of {mo_num} in WO010032...")
                cursor.execute("SELECT MANUFACTUREORDERST_I FROM WO010032 WHERE MANUFACTUREORDER_I = ?", mo_num)
                mo_row = cursor.fetchone()
                if mo_row:
                    print(f"  Status: {mo_row.MANUFACTUREORDERST_I}")
                else:
                    print("  MO not found in WO010032 (maybe in History MOP30100?)")
                    
        else:
            print("  No matches in MOP10213.")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_good_mo()
