import pyodbc
from secrets_loader import build_connection_string

def dump_goldca00():
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        
        item_nmbr = "GOLDCA00"
        print(f"Dumping IV00101 row for {item_nmbr}...")
        
        cursor.execute("SELECT * FROM IV00101 WHERE ITEMNMBR = ?", (item_nmbr,))
        columns = [column[0] for column in cursor.description]
        row = cursor.fetchone()
        
        if row:
            for i, col in enumerate(columns):
                val = row[i]
                # Highlight potential boolean/flag values
                marker = ""
                if str(val) in ["1", "True", "Yes", "1.00000", "1.0"]:
                    marker = " <--- POSSIBLE MATCH (1/True)"
                
                print(f"{col}: {val}{marker}")
        else:
            print(f"Item {item_nmbr} not found!")

        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dump_goldca00()
