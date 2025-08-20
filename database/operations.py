# database/operations.py
"""
Database operations for SQL Bot and Dormant Account Analyzer.
FIXED: All functions now use the correct SQL parameter style for both
SQLAlchemy (:name) and raw pymssql (%(name)s) to prevent syntax errors.
"""
from datetime import datetime
import pandas as pd
import streamlit as st
from sqlalchemy import text
from database.connection import get_db_connection


def save_summary_to_db(observation, trend, insight, action):
    """Saves analysis summary to the insight log table."""
    engine = get_db_connection()
    if engine is None:
        st.error("Cannot save summary: Database connection failed.")
        return False

    try:
        params = {
            'obs': str(observation)[:8000], 'trend': str(trend)[:8000],
            'insight': str(insight)[:8000], 'action': str(action)[:8000]
        }

        # FIX: Use the correct parameter style for each connection type
        if hasattr(engine, 'connect'):  # SQLAlchemy Engine path (uses :name)
            query = text("""
                INSERT INTO insight_log (observation, trend, insight, action)
                VALUES (:obs, :trend, :insight, :action)
            """)
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(query, params)
        else:  # Raw pymssql connection path (uses %(name)s)
            query_str = """
                INSERT INTO insight_log (observation, trend, insight, action)
                VALUES (%(obs)s, %(trend)s, %(insight)s, %(action)s)
            """
            with engine.cursor() as cursor:
                cursor.execute(query_str, params)
            engine.commit()

        return True
    except Exception as e:
        if not hasattr(engine, 'connect'):
            try: engine.rollback()
            except: pass
        st.error(f"Failed to save summary to DB: {e}")
        return False


def save_sql_query_to_history(nl_query, sql_query):
    """Saves an NL query and its SQL translation to the history table."""
    engine = get_db_connection()
    if engine is None: return False

    try:
        params = {'nl_query': nl_query, 'sql_query': sql_query}

        # FIX: Use the correct parameter style for each connection type
        if hasattr(engine, 'connect'):
            query = text("INSERT INTO sql_query_history (natural_language_query, sql_query) VALUES (:nl_query, :sql_query)")
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(query, params)
        else:
            query_str = "INSERT INTO sql_query_history (natural_language_query, sql_query) VALUES (%(nl_query)s, %(sql_query)s)"
            with engine.cursor() as cursor:
                cursor.execute(query_str, params)
            engine.commit()
        return True
    except Exception as e:
        if not hasattr(engine, 'connect'):
            try: engine.rollback()
            except: pass
        st.warning(f"Failed to save query to history: {e}")
        return False


def log_flag_instructions(account_ids, flag_instruction, days_threshold=None):
    """Logs flag instructions to the dormant_flags table."""
    if not isinstance(account_ids, list) or not account_ids:
        return False, "No account IDs provided."

    engine = get_db_connection()
    if engine is None: return False, "Database connection failed."

    try:
        data_to_insert = [{'acc_id': acc_id, 'instr': flag_instruction, 'reason': f"Automated flagging via {flag_instruction}", 'days': days_threshold} for acc_id in account_ids]

        # FIX: Use the correct parameter style for each connection type
        if hasattr(engine, 'connect'):
            query = text("""
                INSERT INTO dormant_flags (account_id, flag_instruction, flag_reason, flag_days, flagged_by)
                VALUES (:acc_id, :instr, :reason, :days, 'system')
            """)
            with engine.connect() as conn:
                with conn.begin():
                    conn.execute(query, data_to_insert)
        else:
            query_str = """
                INSERT INTO dormant_flags (account_id, flag_instruction, flag_reason, flag_days, flagged_by)
                VALUES (%(acc_id)s, %(instr)s, %(reason)s, %(days)s, 'system')
            """
            with engine.cursor() as cursor:
                # pymssql's executemany is optimized for this
                cursor.executemany(query_str, data_to_insert)
            engine.commit()

        return True, f"Successfully logged {len(account_ids)} account flags."
    except Exception as e:
        if not hasattr(engine, 'connect'):
            try: engine.rollback()
            except: pass
        return False, f"Error logging flags: {str(e)}"

# --- Functions below are mostly SELECTs and do not need parameter style changes ---

def save_to_db(df, table_name="accounts_data"):
    if df is None or df.empty: return False
    engine = get_db_connection()
    if engine is None: return False
    try:
        # pd.to_sql is smart enough to handle both connection types, so this function is okay.
        df.to_sql(table_name, engine, if_exists='replace', index=False, chunksize=1000)
        st.sidebar.success(f"Successfully saved {len(df)} rows to {table_name}")
        return True
    except Exception as e:
        st.sidebar.error(f"Database save error for '{table_name}': {e}")
        return False


def prepare_dataframe_for_save(df, db_columns):
    df_prepared = pd.DataFrame()
    df_cols_lower = {col.lower(): col for col in df.columns}
    for db_col in db_columns:
        df_col_name = df_cols_lower.get(db_col.lower())
        if df_col_name:
            df_prepared[db_col] = df[df_col_name]
    if df_prepared.empty:
        st.sidebar.error("No matching columns found between DataFrame and database table.")
        return None
    return clean_dataframe_for_database(df_prepared)


def clean_dataframe_for_database(df):
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df_clean[col] = pd.to_datetime(df_clean[col]).dt.strftime('%Y-%m-%d %H:%M:%S.%f').where(pd.notna(df_clean[col]), None)
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
    return df_clean


def get_recent_sql_history(limit=10):
    engine = get_db_connection()
    if engine is None: return None
    try:
        query = f"SELECT TOP {int(limit)} * FROM sql_query_history ORDER BY timestamp DESC"
        return pd.read_sql(query, engine)
    except Exception as e:
        st.error(f"Error retrieving query history: {e}")
        return None


def execute_custom_query(sql_query: str):
    if not sql_query or not sql_query.strip(): return None
    engine = get_db_connection()
    if engine is None: return None
    try:
        return pd.read_sql(sql_query, engine)
    except Exception as e:
        st.error(f"SQL Query Error: {e}")
        return None


def get_table_data(table_name="accounts_data"):
    engine = get_db_connection()
    if engine is None: return None
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        st.sidebar.success(f"Successfully loaded {len(df)} rows from '{table_name}'.")
        return df
    except Exception as e:
        st.error(f"Error loading data from table '{table_name}': {e}")
        return None


def get_table_count(table_name):
    conn = get_db_connection()
    if conn is None: return 0
    try:
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.fetchone()[0]
        else:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cursor.fetchone()[0]
    except Exception as e:
        st.error(f"Error getting count for {table_name}: {e}")
        return 0


def check_database_health():
    conn = get_db_connection()
    if conn is None: return {"status": "unhealthy", "message": "No database connection"}
    try:
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(text("SELECT GETDATE(), @@VERSION"))
                server_info = result.fetchone()
        else:
            with conn.cursor() as cursor:
                cursor.execute("SELECT GETDATE(), @@VERSION")
                server_info = cursor.fetchone()

        required_tables = ['accounts_data', 'dormant_flags', 'dormant_ledger', 'insight_log', 'sql_query_history']
        table_status = {table: get_table_count(table) for table in required_tables}

        return {"status": "healthy", "server_time": server_info[0], "server_version": server_info[1][:100], "tables": table_status}
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}