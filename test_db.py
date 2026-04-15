
import pyodbc
from secrets_loader import build_connection_string

try:
    print("Building connection string...")
    conn_str, server, db, auth = build_connection_string()
    print(f"Target: {server} / {db} ({auth})")
    
    # Mask password for safety in output
    safe_conn_str = conn_str
    if "PWD=" in safe_conn_str:
        safe_conn_str = safe_conn_str.split("PWD=")[0] + "PWD=*****;" + safe_conn_str.split("PWD=")[1].split(";", 1)[1]
    
    print(f"Connection String: {safe_conn_str}")

    print("Connecting...")
    conn = pyodbc.connect(conn_str)
    print("Connected!")
    
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 ITEMNMBR FROM IV00101")
    row = cursor.fetchone()
    print(f"Test Query Result: {row}")
    
    conn.close()
    print("Connection closed.")

except Exception as e:
    print("FAILED.")
    # Safe print
    try:
        print(f"Error: {str(e)}")
    except:
        print("Error: [Unprintable Exception]")
