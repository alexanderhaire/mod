import pandas as pd
import pyodbc
from secrets_loader import build_connection_string

def debug_pop_minimal():
    print("--- DIAGNOSTIC: POP Minimal Inspection ---")
    
    conn_str, _, _, _ = build_connection_string()
    try:
        conn = pyodbc.connect(conn_str)
        
        # 1. Inspect POP30310 Top 1
        print("--- POP30310 (Line) Top 1 ---")
        df_line = pd.read_sql("SELECT TOP 1 * FROM POP30310", conn)
        print(df_line.to_string())
        
        # 2. Inspect POP30300 (Header) Top 1
        print("\n--- POP30300 (Header) Top 1 ---")
        df_head = pd.read_sql("SELECT TOP 1 * FROM POP30300", conn)
        print(df_head.to_string())

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals(): conn.close()

if __name__ == "__main__":
    debug_pop_minimal()
