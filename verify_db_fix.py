
import pyodbc
from secrets_loader import build_connection_string
from constants import RAW_MATERIAL_CLASS_CODES

def verify_db_classification():
    conn_str, _, _, _ = build_connection_string()
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
    
    # 1. Verify NPK3011 Class
    cursor.execute("SELECT ITEMNMBR, ITMCLSCD FROM IV00101 WHERE ITEMNMBR = 'NPK3011'")
    row = cursor.fetchone()
    if not row:
        print("NPK3011 not found in IV00101")
        return

    item_class = row.ITMCLSCD.strip()
    print(f"NPK3011 Item Class: '{item_class}'")
    
    # 2. Verify Logic
    is_raw_material = item_class in RAW_MATERIAL_CLASS_CODES
    print(f"Is '{item_class}' in RAW_MATERIAL_CLASS_CODES? {is_raw_material}")
    
    # 3. Simulate App Query Logic
    raw_mat_list = "', '".join(RAW_MATERIAL_CLASS_CODES)
    query = f"""
    SELECT ITEMNMBR 
    FROM IV00101 i
    WHERE ITEMNMBR = 'NPK3011'
      AND UPPER(LTRIM(RTRIM(i.ITMCLSCD))) IN ('{raw_mat_list}')
    """
    cursor.execute(query)
    found = cursor.fetchone()
    print(f"Found via SQL Filter? {'YES' if found else 'NO'}")

if __name__ == "__main__":
    verify_db_classification()
