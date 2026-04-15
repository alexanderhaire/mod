import pyodbc
from secrets_loader import build_connection_string

def create_overrides_table():
    conn_str, _, _, _ = build_connection_string()
    
    create_table_sql = """
    IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[dbo].[PURCHASING_MOD_OVERRIDES]') AND type in (N'U'))
    BEGIN
        CREATE TABLE [dbo].[PURCHASING_MOD_OVERRIDES](
            [ITEMNMBR] [char](31) NOT NULL,
            [LOCKED_COST] [numeric](19, 5) NOT NULL,
            [MODIFIED_BY] [varchar](50) NULL,
            [MODIFIED_DATE] [datetime] NULL,
            CONSTRAINT [PK_PURCHASING_MOD_OVERRIDES] PRIMARY KEY CLUSTERED 
            (
                [ITEMNMBR] ASC
            )
        )
    END
    """
    
    try:
        conn = pyodbc.connect(conn_str, autocommit=True)
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        print("Table PURCHASING_MOD_OVERRIDES created or already exists.")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_overrides_table()
