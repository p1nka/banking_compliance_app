# --- START OF FILE operations.py ---

# database/operations.py
"""
Database operations for SQL Bot and Dormant Account Analyzer.
Enhanced for Azure SQL Database compatibility using SQLAlchemy.
"""
from datetime import datetime

import pandas as pd
import streamlit as st
from sqlalchemy import text
from database.connection import get_db_connection


def save_to_db(df, table_name="accounts_data"):
    """Saves DataFrame to Azure SQL Database, replacing table data."""
    if df is None or df.empty:
        st.sidebar.warning(f"Skipped saving empty DataFrame to '{table_name}'.")
        return False

    engine = get_db_connection()
    if engine is None:
        st.sidebar.error(f"Cannot save: No database connection for '{table_name}'.")
        return False

    try:
        with engine.connect() as conn:
            with conn.begin() as trans:
                # Get database columns to ensure alignment
                result = conn.execute(
                    text(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = :table"),
                    {'table': table_name})
                db_columns = [row[0] for row in result]

                # Prepare DataFrame for saving
                df_prepared = prepare_dataframe_for_save(df, db_columns)
                if df_prepared is None:
                    return False

                # Clear existing data and insert new data
                conn.execute(text(f"TRUNCATE TABLE {table_name}"))
                df_prepared.to_sql(table_name, conn, if_exists='append', index=False, chunksize=1000, method='multi')
                trans.commit()
                st.sidebar.success(f"Successfully saved {len(df_prepared)} rows to {table_name}")
                return True
    except Exception as e:
        st.sidebar.error(f"Database save error for '{table_name}': {e}")
        import traceback
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        return False


def prepare_dataframe_for_save(df, db_columns):
    """Prepares a DataFrame for saving by aligning and cleaning columns."""
    df_prepared = pd.DataFrame()
    df_cols_lower = {col.lower(): col for col in df.columns}

    # Align DataFrame columns with database columns (case-insensitive)
    for db_col in db_columns:
        df_col_name = df_cols_lower.get(db_col.lower())
        if df_col_name:
            df_prepared[db_col] = df[df_col_name]

    if df_prepared.empty:
        st.sidebar.error("No matching columns found between DataFrame and database table.")
        return None

    return clean_dataframe_for_database(df_prepared)


def clean_dataframe_for_database(df):
    """Cleans DataFrame data types for database compatibility."""
    df_clean = df.copy()

    for col in df_clean.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        # NOTE: Converting datetimes to strings is a workaround for some driver issues.
        # to_sql often handles native datetime objects well, but this is safer.
        df_clean[col] = pd.to_datetime(df_clean[col]).dt.strftime('%Y-%m-%d %H:%M:%S.%f').where(pd.notna(df_clean[col]),
                                                                                                None)

    for col in df_clean.select_dtypes(include=['object']).columns:
        # Replace NaN with None which to_sql handles as NULL
        df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)

    return df_clean


def save_summary_to_db(observation, trend, insight, action):
    """Saves analysis summary to the insight log table."""
    engine = get_db_connection()
    if engine is None:
        st.error("Cannot save summary: Database connection failed.")
        return False

    try:
        # FIX: Use text() for the SQL string and named parameters
        query = text("""
            INSERT INTO insight_log (observation, trend, insight, action)
            VALUES (:obs, :trend, :insight, :action)
        """)
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(query, {
                    'obs': str(observation)[:8000],
                    'trend': str(trend)[:8000],
                    'insight': str(insight)[:8000],
                    'action': str(action)[:8000]
                })
        return True
    except Exception as e:
        st.error(f"Failed to save summary to DB: {e}")
        return False


def save_sql_query_to_history(nl_query, sql_query):
    """Saves an NL query and its SQL translation to the history table."""
    engine = get_db_connection()
    if engine is None:
        st.warning("Cannot save to history: Database connection failed.")
        return False
    try:
        # FIX: Use text() for the SQL string and named parameters
        query = text("""
            INSERT INTO sql_query_history (natural_language_query, sql_query)
            VALUES (:nl_query, :sql_query)
        """)
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(query, {'nl_query': nl_query, 'sql_query': sql_query})
        return True
    except Exception as e:
        st.warning(f"Failed to save query to history: {e}")
        return False


def get_recent_sql_history(limit=10):
    """Retrieves recent SQL query history from the database."""
    engine = get_db_connection()
    if engine is None: return None
    try:
        # Using an f-string for TOP is generally safe if `limit` is not from user input.
        query = f"SELECT TOP {int(limit)} * FROM sql_query_history ORDER BY timestamp DESC"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error retrieving query history: {e}")
        return None


def log_flag_instructions(account_ids, flag_instruction, days_threshold=None):
    """Logs flag instructions to the dormant_flags table."""
    if not isinstance(account_ids, list) or not account_ids:
        return False, "No account IDs provided for flagging."

    engine = get_db_connection()
    if engine is None:
        return False, "Database connection failed."

    try:
        # FIX: Use text() for the SQL string and named parameters
        query = text("""
            INSERT INTO dormant_flags (account_id, flag_instruction, flag_reason, flag_days, flagged_by)
            VALUES (:acc_id, :instr, :reason, :days, 'system')
        """)

        data_to_insert = [
            {
                'acc_id': acc_id,
                'instr': flag_instruction,
                'reason': f"Automated flagging via {flag_instruction}",
                'days': days_threshold
            }
            for acc_id in account_ids
        ]

        with engine.connect() as conn:
            with conn.begin():
                conn.execute(query, data_to_insert)

        return True, f"Successfully logged {len(account_ids)} account flags."
    except Exception as e:
        return False, f"Error logging flags: {str(e)}"

# In database/operations.py, ADD THIS ENTIRE FUNCTION

def execute_custom_query(sql_query: str):
    """Executes a custom SQL query and returns the result as a DataFrame."""
    if not sql_query or not sql_query.strip():
        st.warning("Cannot execute an empty SQL query.")
        return None

    engine = get_db_connection()
    if engine is None:
        st.error("Cannot execute query: Database connection failed.")
        return None

    try:
        # Use pandas to directly execute the query and fetch results
        df = pd.read_sql(sql_query, engine)
        return df
    except Exception as e:
        # Provide a more informative error message to the user in the UI
        st.error(f"SQL Query Error: {e}")
        return None


# In database/operations.py, ADD THIS ENTIRE FUNCTION at the end of the file

def get_table_data(table_name="accounts_data"):
    """
    Retrieves all data from a specified table in the database.

    Args:
        table_name (str): The name of the table to fetch data from.

    Returns:
        pd.DataFrame: A DataFrame containing the table data, or None if an error occurs.
    """
    engine = get_db_connection()
    if engine is None:
        st.error(f"Cannot get table data: Database connection failed.")
        return None

    try:
        # Using an f-string is generally safe here if table_name is controlled by the app
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        st.sidebar.success(f"Successfully loaded {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        st.error(f"Error loading data from table '{table_name}': {e}")
        return None


def get_table_count(table_name):
    """Get the number of rows in a table."""
    conn = get_db_connection()
    if conn is None:
        return 0

    try:
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(f"SELECT COUNT(*) FROM {table_name}")
                return result.fetchone()[0]
        else:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
    except Exception as e:
        st.error(f"Error getting count for {table_name}: {e}")
        return 0


def backup_table_data(table_name):
    """Create a backup of table data."""
    try:
        df = get_table_data(table_name, limit=None)  # Get all data
        if df is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{table_name}_backup_{timestamp}.csv"
            df.to_csv(filename, index=False)
            st.success(f"Backup created: {filename}")
            return True
        return False
    except Exception as e:
        st.error(f"Error creating backup for {table_name}: {e}")
        return False


def check_database_health():
    """Check the health of the database connection and tables."""
    conn = get_db_connection()
    if conn is None:
        return {"status": "unhealthy", "message": "No database connection"}

    try:
        # Test basic connectivity
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute("SELECT GETDATE(), @@VERSION")
                server_info = result.fetchone()
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT GETDATE(), @@VERSION")
            server_info = cursor.fetchone()
            cursor.close()

        # Check table existence
        required_tables = ['accounts_data', 'dormant_flags', 'dormant_ledger', 'insight_log', 'sql_query_history']
        table_status = {}

        for table in required_tables:
            count = get_table_count(table)
            table_status[table] = count

        return {
            "status": "healthy",
            "server_time": server_info[0],
            "server_version": server_info[1][:100],  # Truncate version string
            "tables": table_status
        }

    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}