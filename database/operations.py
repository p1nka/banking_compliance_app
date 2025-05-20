import pyodbc
import pandas as pd
import streamlit as st
from datetime import datetime
import time
from .connection import get_db_connection


def save_to_db(df, table_name="accounts_data"):
    """Saves DataFrame to Azure SQL (default connection), replacing table data."""
    if df is None or df.empty:
        st.sidebar.warning(f"Skipped saving empty or None DataFrame to '{table_name}'.")
        return False

    conn = get_db_connection()  # Uses the cached connection
    if conn is None:
        st.sidebar.error(f"Cannot save to DB: Default DB connection failed for '{table_name}'.")
        return False

    try:
        # First, check if the table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{table_name}'")
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
            return False

        # Determine which columns from the DataFrame to save based on the table schema
        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'")
        db_columns = [row[0] for row in cursor.fetchall()]

        # Debug output to help diagnose column issues
        st.sidebar.info(f"Database columns: {', '.join(db_columns)}")
        st.sidebar.info(f"DataFrame columns: {', '.join(df.columns)}")

        # Find matching columns between DataFrame and DB table (case-insensitive)
        cols_to_save = []
        df_cols_lower = [col.lower() for col in df.columns]

        # Create a mapping from lowercase column names to actual column names
        df_col_map = {col.lower(): col for col in df.columns}

        for db_col in db_columns:
            db_col_lower = db_col.lower()
            if db_col_lower in df_cols_lower:
                # Use the actual column name from the DataFrame
                orig_col = df_col_map[db_col_lower]
                cols_to_save.append((db_col, orig_col))  # (DB column name, DataFrame column name)

        if not cols_to_save:
            st.sidebar.error(f"No matching columns found in DataFrame for table '{table_name}'. Cannot save.")
            return False

        st.sidebar.info(f"Matched columns: {', '.join([f'{df_col} -> {db_col}' for db_col, df_col in cols_to_save])}")

        # Prepare DataFrame for saving
        db_cols = [db_col for db_col, _ in cols_to_save]
        df_cols = [df_col for _, df_col in cols_to_save]

        # Make a copy of just the columns we need
        df_to_save = df[df_cols].copy()

        # Convert datetime columns to string format compatible with SQL Server DATETIME2
        for col in df_to_save.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            # Convert NaT to None so pyodbc can handle it as SQL NULL
            df_to_save[col] = df_to_save[col].where(pd.notna(df_to_save[col]), None)
            # Format valid dates
            df_to_save[col] = df_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(x, datetime) else None)

        # Convert Yes/No columns to standardized format for SQL
        boolean_like_columns = [
            'FTD_Auto_Renewal', 'SDB_Tenant_Communication_Received',
            'Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Customer_Responded_to_Bank_Contact',
            'Claim_Successful', 'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
            'Customer_Has_Litigation_Regulatory_Reqs', 'Holder_Has_Activity_On_Any_Other_Account',
            'Is_Asset_Only_Customer_Type', 'Expected_Account_Dormant',
            'Expected_Requires_Article_3_Process', 'Expected_Transfer_to_CB_Due'
        ]

        for df_col in df_cols:
            if df_col in boolean_like_columns or any(col.lower() == df_col.lower() for col in boolean_like_columns):
                # Ensure values are standardized as 'Yes'/'No'/'Unknown'
                df_to_save[df_col] = df_to_save[df_col].astype(str)
                df_to_save[df_col] = df_to_save[df_col].replace({
                    'true': 'Yes', 'True': 'Yes', 'TRUE': 'Yes', 'yes': 'Yes', 'Yes': 'Yes', 'y': 'Yes', 'Y': 'Yes',
                    'false': 'No', 'False': 'No', 'FALSE': 'No', 'no': 'No', 'No': 'No', 'n': 'No', 'N': 'No',
                    'nan': 'Unknown', 'None': 'Unknown', 'none': 'Unknown', 'unknown': 'Unknown', 'Unknown': 'Unknown'
                })

        # For text columns, replace None/NaN with appropriate text
        text_columns = [col for col in df_to_save.columns if col not in
                        boolean_like_columns and
                        not pd.api.types.is_numeric_dtype(df_to_save[col]) and
                        not pd.api.types.is_datetime64_any_dtype(df_to_save[col])]

        for col in text_columns:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].fillna('Unknown')

        # Truncate DB table and insert data
        with conn.cursor() as cursor:
            # Use a transaction for the operation
            with conn:
                cursor.execute(f"TRUNCATE TABLE {table_name}")

                # Build the INSERT statement template
                db_columns_str = ','.join([f'[{c}]' for c in db_cols])  # Enclose column names in brackets
                placeholders = ','.join(['?'] * len(cols_to_save))
                insert_sql = f"INSERT INTO {table_name} ({db_columns_str}) VALUES ({placeholders})"

                # Prepare values as tuples for executemany
                values_to_insert = []
                for index, row in df_to_save.iterrows():
                    # Convert pandas NaN/NaT to Python None for pyodbc
                    prepared_values = []
                    for df_col in df_cols:
                        value = row[df_col]
                        if pd.isna(value):  # Check for pandas missing values (NaT, NaN)
                            prepared_values.append(None)
                        else:
                            prepared_values.append(value)
                    values_to_insert.append(tuple(prepared_values))

                if values_to_insert:
                    # Use executemany for better performance
                    cursor.executemany(insert_sql, values_to_insert)

        # Sleep briefly to allow the DB connection to stabilize
        time.sleep(1)
        st.sidebar.success(f"Successfully saved {len(df_to_save)} rows to {table_name}")
        return True
    except pyodbc.Error as e:
        st.sidebar.error(f"Database Save Error ('{table_name}'): {e}. Check data compatibility or constraints.")
        return False
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during DB save ('{table_name}'): {e}")
        import traceback
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        return False


def save_summary_to_db(observation, trend, insight, action):
    """Saves analysis summary to the insight log table using default DB connection."""
    conn = get_db_connection()  # Uses the cached connection
    if conn is None:
        st.error("Cannot save summary to DB: Default DB connection failed.")
        return False

    try:
        with conn:
            cursor = conn.cursor()
            insert_sql = """
                         INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                         VALUES (?, ?, ?, ?, ?)
                         """
            timestamp = datetime.now()
            # Ensure data types match DB schema NVARCHAR(MAX)
            cursor.execute(insert_sql, (timestamp, str(observation)[:8000], str(trend)[:8000], str(insight)[:8000],
                                        str(action)[:8000]))  # Truncate to fit MAX size if needed
            conn.commit()
        return True
    except pyodbc.Error as e:
        st.error(f"Failed to save summary to DB: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred saving summary: {e}")
        return False


def save_sql_query_to_history(nl_query, sql_query):
    """Saves an NL query and its SQL translation to the query history table."""
    conn = get_db_connection()
    if conn is None:
        st.warning("Cannot save to history: Default DB connection failed.")
        return False

    try:
        with conn:
            cursor = conn.cursor()
            # Check if table exists
            cursor.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'sql_query_history'")
            exists = cursor.fetchone() is not None

            if not exists:
                st.warning("SQL query history table not found. Query not saved.")
                return False

            # Insert the query into history
            cursor.execute(
                "INSERT INTO sql_query_history (natural_language_query, sql_query) VALUES (?, ?)",
                (nl_query, sql_query)
            )
            conn.commit()
        return True
    except Exception as e:
        st.warning(f"Failed to save query to history: {e}")
        return False


def get_recent_sql_history(limit=10):
    """Retrieves recent SQL query history from the database."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        history_df = pd.read_sql(
            f"SELECT TOP {limit} timestamp, natural_language_query, sql_query FROM sql_query_history ORDER BY timestamp DESC",
            conn
        )
        return history_df
    except Exception as e:
        st.error(f"Error retrieving query history: {e}")
        return None


def log_flag_instructions(account_ids, flag_instruction, days_threshold=None):
    """Log flag instructions to the dormant_flags table."""
    if not account_ids:
        return False, "No account IDs provided for flagging."

    conn = get_db_connection()
    if conn is None:
        return False, "Database connection failed. Could not log flags."

    try:
        success_count = 0
        timestamp = datetime.now()

        with conn:
            cursor = conn.cursor()

            for account_id in account_ids:
                try:
                    cursor.execute(
                        """
                        INSERT INTO dormant_flags
                        (account_id, flag_instruction, flag_reason, flag_days, flagged_by, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            account_id,
                            flag_instruction,
                            f"Automated flagging via {flag_instruction}",
                            days_threshold,
                            "system",
                            timestamp
                        )
                    )
                    success_count += 1
                except Exception:
                    continue

            conn.commit()

        if success_count == len(account_ids):
            return True, f"Successfully logged all {success_count} account flags."
        elif success_count > 0:
            return True, f"Partially successful: Logged {success_count} of {len(account_ids)} account flags."
        else:
            return False, "Failed to log any account flags."

    except Exception as e:
        return False, f"Error logging flags: {str(e)}"


# database/operations.py
"""
Database operations for SQL Bot and Dormant Account Analyzer.
"""

import pandas as pd
import json
from datetime import datetime
from database.connection import get_db_connection


def save_sql_query_to_history(natural_language_query, sql_query):
    """
    Save a SQL query and its natural language equivalent to history.

    Args:
        natural_language_query (str): The original natural language question
        sql_query (str): The generated SQL query

    Returns:
        bool: True if the save was successful, False otherwise
    """
    try:
        conn = get_db_connection()
        if not conn:
            return False

        cursor = conn.cursor()

        # Create history table if it doesn't exist
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'sql_query_history')
        BEGIN
            CREATE TABLE sql_query_history (
                id INT IDENTITY(1,1) PRIMARY KEY,
                natural_language_query NVARCHAR(MAX),
                sql_query NVARCHAR(MAX),
                timestamp DATETIME DEFAULT GETDATE()
            )
        END
        """)

        # Insert the new query
        cursor.execute("""
                       INSERT INTO sql_query_history (natural_language_query, sql_query)
                       VALUES (?, ?)
                       """, (natural_language_query, sql_query))

        conn.commit()
        cursor.close()

        return True
    except Exception as e:
        print(f"Error saving SQL query to history: {e}")
        return False


def get_recent_sql_history(limit=10):
    """
    Get the most recent SQL queries from history.

    Args:
        limit (int): Maximum number of queries to return

    Returns:
        pandas.DataFrame: DataFrame containing the history, or None if error
    """
    try:
        conn = get_db_connection()
        if not conn:
            return None

        # Query the history table
        query = f"""
        SELECT TOP {limit} id, natural_language_query, sql_query, timestamp
        FROM sql_query_history
        ORDER BY timestamp DESC
        """

        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error retrieving SQL history: {e}")
        return None


def save_dormant_analysis_to_history(parameters, count):
    """
    Save dormant account analysis parameters and results to history.

    Args:
        parameters (dict): The analysis parameters
        count (int): Number of dormant accounts found

    Returns:
        bool: True if the save was successful, False otherwise
    """
    try:
        conn = get_db_connection()
        if not conn:
            return False

        cursor = conn.cursor()

        # Create history table if it doesn't exist
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'dormant_analysis_history')
        BEGIN
            CREATE TABLE dormant_analysis_history (
                id INT IDENTITY(1,1) PRIMARY KEY,
                parameters NVARCHAR(MAX),
                count INT,
                timestamp DATETIME DEFAULT GETDATE()
            )
        END
        """)

        # Convert parameters dict to JSON string
        params_json = json.dumps(parameters)

        # Insert the new analysis
        cursor.execute("""
                       INSERT INTO dormant_analysis_history (parameters, count)
                       VALUES (?, ?)
                       """, (params_json, count))

        conn.commit()
        cursor.close()

        return True
    except Exception as e:
        print(f"Error saving dormant analysis to history: {e}")
        return False


def get_recent_dormant_analysis(limit=10):
    """
    Get the most recent dormant account analyses from history.

    Args:
        limit (int): Maximum number of analyses to return

    Returns:
        pandas.DataFrame: DataFrame containing the history, or None if error
    """
    try:
        conn = get_db_connection()
        if not conn:
            return None

        # Query the history table
        query = f"""
        SELECT TOP {limit} id, parameters, count, timestamp
        FROM dormant_analysis_history
        ORDER BY timestamp DESC
        """

        df = pd.read_sql(query, conn)
        return df
    except Exception as e:
        print(f"Error retrieving dormant analysis history: {e}")
        return None

# Add any additional database operations needed for your application here