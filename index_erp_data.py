"""
A script to index data from the ERP system into the vector store.
This is intended to be run as a one-off script to build the index.
"""

import pyodbc
from pathlib import Path

from constants import LOGGER
from secrets_loader import build_connection_string
from vector_store import VectorStore

def index_item_descriptions(cursor: pyodbc.Cursor, vector_store: VectorStore):
    """Index item descriptions from the IV00101 table."""
    LOGGER.info("Indexing item descriptions from IV00101...")
    
    query = "SELECT ITEMNMBR, ITEMDESC FROM IV00101 WHERE ITEMDESC IS NOT NULL AND ITEMDESC != ''"
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        texts = []
        metadatas = []
        
        for row in rows:
            text = f"Item: {row.ITEMNMBR}, Description: {row.ITEMDESC}"
            texts.append(text)
            metadatas.append({"source": "IV00101", "item_number": row.ITEMNMBR})
            
        vector_store.add(texts, metadatas)
        LOGGER.info(f"Successfully indexed {len(texts)} item descriptions.")
        
    except pyodbc.Error as err:
        LOGGER.error(f"Failed to fetch item descriptions for indexing: {err}")

def index_customer_names(cursor: pyodbc.Cursor, vector_store: VectorStore):
    """Index customer names from the RM00101 table."""
    LOGGER.info("Indexing customer names from RM00101...")
    
    query = "SELECT CUSTNMBR, CUSTNAME FROM RM00101 WHERE CUSTNAME IS NOT NULL AND CUSTNAME != ''"
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        texts = []
        metadatas = []
        
        for row in rows:
            text = f"Customer: {row.CUSTNMBR}, Name: {row.CUSTNAME}"
            texts.append(text)
            metadatas.append({"source": "RM00101", "customer_number": row.CUSTNMBR})
            
        vector_store.add(texts, metadatas)
        LOGGER.info(f"Successfully indexed {len(texts)} customer names.")
        
    except pyodbc.Error as err:
        LOGGER.error(f"Failed to fetch customer names for indexing: {err}")

def index_vendor_names(cursor: pyodbc.Cursor, vector_store: VectorStore):
    """Index vendor names from the PM00200 table."""
    LOGGER.info("Indexing vendor names from PM00200...")
    
    query = "SELECT VENDORID, VENDNAME FROM PM00200 WHERE VENDNAME IS NOT NULL AND VENDNAME != ''"
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        
        texts = []
        metadatas = []
        
        for row in rows:
            text = f"Vendor: {row.VENDORID}, Name: {row.VENDNAME}"
            texts.append(text)
            metadatas.append({"source": "PM00200", "vendor_id": row.VENDORID})
            
        vector_store.add(texts, metadatas)
        LOGGER.info(f"Successfully indexed {len(texts)} vendor names.")
        
    except pyodbc.Error as err:
        LOGGER.error(f"Failed to fetch vendor names for indexing: {err}")

def main():
    """Main function to run the indexing process."""
    LOGGER.info("Starting ERP data indexing...")
    
    vector_store_path = Path("erp_vector_store.json")
    vector_store = VectorStore(vector_store_path)
    
    try:
        conn_str, _, _, _ = build_connection_string()
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            index_item_descriptions(cursor, vector_store)
            index_customer_names(cursor, vector_store)
            index_vendor_names(cursor, vector_store)
            
    except (pyodbc.Error, RuntimeError, KeyError) as err:
        LOGGER.error(f"Failed to connect to the database for indexing: {err}")
        
    LOGGER.info("ERP data indexing complete.")

if __name__ == "__main__":
    main()
