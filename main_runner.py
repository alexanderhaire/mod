import datetime
import logging

import pyodbc

from constants import LOGGER
from router import handle_question
from secrets_loader import build_connection_string


def main():
    """Example of how to use the reasoning core."""
    logging.basicConfig(level=logging.INFO)
    
    # This example assumes you have a secrets.toml file configured.
    # Example secrets.toml:
    # [sql]
    # server = "your_server"
    # database = "your_db"
    # authentication = "windows" # or "sql"
    # # for sql auth:
    # # username = "your_user"
    # # password = "your_password"
    #
    # [openai]
    # api_key = "your_openai_key"

    prompt = "What was the usage for NO3CA12 last month?"
    today = datetime.date.today()
    
    try:
        conn_str, _, _, _ = build_connection_string()
    except (RuntimeError, KeyError) as err:
        LOGGER.error(f"Error building connection string: {err}. Please check your secrets.toml.")
        return

    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            
            # In a real app, context would be built from chat history
            context = {"intent": "report", "item": "NO3CA12"}
            
            print(f"Executing prompt: '{prompt}'")
            sql_payload = handle_question(cursor, prompt, today, context)

            if sql_payload.get("error"):
                print(f"\n--- Error ---\n{sql_payload['error']}")
                if sql := sql_payload.get("sql"):
                    print(f"\n--- Faulty SQL ---\n{sql}")
            else:
                insights = sql_payload.get("insights") or {}
                print(f"\n--- Summary ---\n{insights.get('summary')}")
                if sql_payload.get("sql"):
                    print(f"\n--- Generated SQL ---\n{sql_payload.get('sql')}")
                row_count = insights.get("row_count")
                if row_count is None:
                    row_count = len(sql_payload.get("data", []))
                print(f"\n--- Data ({row_count} rows) ---")
                for row in sql_payload.get("data", []):
                    print(row)

    except pyodbc.Error as err:
        LOGGER.error(f"Database connection error: {err}")
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred: {e}")
