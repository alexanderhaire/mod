
import pyodbc
import os

def check_columns():
    # Use trusted connection as seen in app.py
    try:
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=AGBAPMG02;"
            "DATABASE=GPCG;"
            "Trusted_Connection=yes;"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 * FROM PM00200")
        columns = [column[0] for column in cursor.description]
        print(f"Columns: {columns}")
        
        # Check specific address columns
        required = ['ADDRESS1', 'CITY', 'STATE', 'ZIPCODE']
        found = [c for c in columns if c in required]
        print(f"Found required: {found}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_columns()
