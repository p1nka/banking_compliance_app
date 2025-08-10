# database/operations.py
"""
Database operations for SQL Bot and Dormant Account Analyzer.
Enhanced for Azure SQL Database compatibility.
"""

import pandas as pd
import streamlit as st
from datetime import datetime
import time
import json
from database.connection import get_db_connection


def save_to_db(df, table_name="accounts_data"):
    """
    Saves DataFrame to Azure SQL Database, replacing table data.
    Enhanced with better error handling and Azure SQL compatibility.
    """
    if df is None or df.empty:
        st.sidebar.warning(f"Skipped saving empty or None DataFrame to '{table_name}'.")
        return False

    conn = get_db_connection()
    if conn is None:
        st.sidebar.error(f"Cannot save to DB: Database connection failed for '{table_name}'.")
        return False

    try:
        # Handle different connection types
        if hasattr(conn, 'connect'):
            # SQLAlchemy engine
            return save_to_db_sqlalchemy_engine(df, table_name, conn)
        elif hasattr(conn, 'execute'):
            # SQLAlchemy connection
            return save_to_db_sqlalchemy_connection(df, table_name, conn)
        else:
            # Direct connection (pymssql)
            return save_to_db_direct(df, table_name, conn)

    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during DB save ('{table_name}'): {e}")
        import traceback
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        return False


def save_to_db_sqlalchemy_engine(df, table_name, engine):
    """Save DataFrame using SQLAlchemy engine."""
    try:
        with engine.connect() as conn:
            # Check if table exists
            result = conn.execute(f"""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
            """)
            table_exists = result.fetchone()[0] > 0

            if not table_exists:
                st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
                return False

            # Prepare DataFrame for saving
            df_prepared = prepare_dataframe_for_save(df, table_name, conn)
            if df_prepared is None:
                return False

            # Clear existing data and insert new data
            trans = conn.begin()
            try:
                conn.execute(f"TRUNCATE TABLE {table_name}")
                df_prepared.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
                trans.commit()

                st.sidebar.success(f"Successfully saved {len(df_prepared)} rows to {table_name}")
                return True
            except Exception as e:
                trans.rollback()
                raise e

    except Exception as e:
        st.sidebar.error(f"SQLAlchemy engine save error: {e}")
        return False


def save_to_db_sqlalchemy_connection(df, table_name, conn):
    """Save DataFrame using SQLAlchemy connection."""
    try:
        # Check if table exists
        result = conn.execute(f"""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
        """)
        table_exists = result.fetchone()[0] > 0

        if not table_exists:
            st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
            return False

        # Prepare DataFrame for saving
        df_prepared = prepare_dataframe_for_save(df, table_name, conn)
        if df_prepared is None:
            return False

        # Clear existing data and insert new data
        conn.execute(f"TRUNCATE TABLE {table_name}")
        df_prepared.to_sql(table_name, conn, if_exists='append', index=False, method='multi')

        st.sidebar.success(f"Successfully saved {len(df_prepared)} rows to {table_name}")
        return True

    except Exception as e:
        st.sidebar.error(f"SQLAlchemy connection save error: {e}")
        return False


def save_to_db_direct(df, table_name, conn):
    """Save DataFrame using direct connection."""
    try:
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(f"""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
        """)
        table_exists = cursor.fetchone()[0] > 0

        if not table_exists:
            st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
            cursor.close()
            return False

        # Get database columns
        cursor.execute(f"""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
            ORDER BY ORDINAL_POSITION
        """)
        db_columns = [row[0] for row in cursor.fetchall()]

        # Match DataFrame columns to database columns
        matched_columns = match_dataframe_columns(df, db_columns)
        if not matched_columns:
            st.sidebar.error(f"No matching columns found between DataFrame and table '{table_name}'.")
            cursor.close()
            return False

        # Prepare DataFrame
        df_prepared = prepare_dataframe_for_direct_save(df, matched_columns)

        # Clear existing data
        cursor.execute(f"TRUNCATE TABLE {table_name}")

        # Insert new data
        if len(df_prepared) > 0:
            db_cols = [col[0] for col in matched_columns]  # Database column names
            df_cols = [col[1] for col in matched_columns]  # DataFrame column names

            # Build INSERT statement
            columns_str = ','.join([f'[{col}]' for col in db_cols])
            placeholders = ','.join(['?' for _ in db_cols])
            insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            # Prepare data for batch insert
            values_to_insert = []
            for _, row in df_prepared.iterrows():
                row_values = []
                for df_col in df_cols:
                    value = row[df_col]
                    if pd.isna(value):
                        row_values.append(None)
                    else:
                        row_values.append(value)
                values_to_insert.append(tuple(row_values))

            # Execute batch insert
            if values_to_insert:
                cursor.executemany(insert_sql, values_to_insert)

        conn.commit()
        cursor.close()

        st.sidebar.success(f"Successfully saved {len(df_prepared)} rows to {table_name}")
        return True

    except Exception as e:
        st.sidebar.error(f"Direct connection save error: {e}")
        return False


def prepare_dataframe_for_save(df, table_name, conn):
    """Prepare DataFrame for saving to database."""
    try:
        # Get database columns
        result = conn.execute(f"""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'dbo'
            ORDER BY ORDINAL_POSITION
        """)
        db_columns = [row[0] for row in result]

        # Match columns
        matched_columns = match_dataframe_columns(df, db_columns)
        if not matched_columns:
            st.sidebar.error(f"No matching columns found for table '{table_name}'.")
            return None

        # Create DataFrame with only matched columns
        df_cols = [col[1] for col in matched_columns]  # DataFrame column names
        db_cols = [col[0] for col in matched_columns]  # Database column names

        df_prepared = df[df_cols].copy()
        df_prepared.columns = db_cols  # Rename to match database

        # Clean up data types
        df_prepared = clean_dataframe_for_database(df_prepared)

        return df_prepared

    except Exception as e:
        st.sidebar.error(f"Error preparing DataFrame: {e}")
        return None


def prepare_dataframe_for_direct_save(df, matched_columns):
    """Prepare DataFrame for direct connection save."""
    df_cols = [col[1] for col in matched_columns]
    df_prepared = df[df_cols].copy()

    # Clean up data types
    df_prepared = clean_dataframe_for_database(df_prepared)

    return df_prepared


def match_dataframe_columns(df, db_columns):
    """Match DataFrame columns to database columns (case-insensitive)."""
    matched_columns = []
    df_cols_lower = {col.lower(): col for col in df.columns}

    for db_col in db_columns:
        db_col_lower = db_col.lower()
        if db_col_lower in df_cols_lower:
            matched_columns.append((db_col, df_cols_lower[db_col_lower]))

    return matched_columns


def clean_dataframe_for_database(df):
    """Clean DataFrame data types for database compatibility."""
    df_clean = df.copy()

    # Handle datetime columns
    for col in df_clean.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
        df_clean[col] = df_clean[col].where(pd.notna(df_clean[col]), None)
        # Convert to string format that SQL Server can handle
        df_clean[col] = df_clean[col].apply(
            lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else None
        )

    # Handle boolean-like columns
    boolean_columns = [
        'FTD_Auto_Renewal', 'SDB_Tenant_Communication_Received',
        'Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Customer_Responded_to_Bank_Contact',
        'Claim_Successful', 'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
        'Customer_Has_Litigation_Regulatory_Reqs', 'Holder_Has_Activity_On_Any_Other_Account',
        'Is_Asset_Only_Customer_Type', 'Expected_Account_Dormant',
        'Expected_Requires_Article_3_Process', 'Expected_Transfer_to_CB_Due'
    ]

    for col in df_clean.columns:
        if col in boolean_columns or any(bool_col.lower() == col.lower() for bool_col in boolean_columns):
            df_clean[col] = df_clean[col].astype(str).replace({
                'true': 'Yes', 'True': 'Yes', 'TRUE': 'Yes', 'yes': 'Yes', 'y': 'Yes', 'Y': 'Yes',
                'false': 'No', 'False': 'No', 'FALSE': 'No', 'no': 'No', 'n': 'No', 'N': 'No',
                'nan': 'Unknown', 'None': 'Unknown', 'none': 'Unknown'
            })

    # Handle text columns - replace NaN with appropriate defaults
    for col in df_clean.select_dtypes(include=['object']).columns:
        if col not in boolean_columns:
            df_clean[col] = df_clean[col].fillna('Unknown')

    # Handle numeric columns - ensure proper types
    for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    return df_clean


def save_summary_to_db(observation, trend, insight, action):
    """Saves analysis summary to the insight log table."""
    conn = get_db_connection()
    if conn is None:
        st.error("Cannot save summary to DB: Database connection failed.")
        return False

    try:
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                connection.execute("""
                    INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                    VALUES (GETDATE(), ?, ?, ?, ?)
                """, (str(observation)[:8000], str(trend)[:8000], str(insight)[:8000], str(action)[:8000]))
        elif hasattr(conn, 'execute'):
            conn.execute("""
                INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                VALUES (GETDATE(), ?, ?, ?, ?)
            """, (str(observation)[:8000], str(trend)[:8000], str(insight)[:8000], str(action)[:8000]))
        else:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                VALUES (GETDATE(), ?, ?, ?, ?)
            """, (str(observation)[:8000], str(trend)[:8000], str(insight)[:8000], str(action)[:8000]))
            conn.commit()
            cursor.close()

        return True
    except Exception as e:
        st.error(f"Failed to save summary to DB: {e}")
        return False


def save_sql_query_to_history(nl_query, sql_query):
    """Saves an NL query and its SQL translation to the query history table."""
    conn = get_db_connection()
    if conn is None:
        st.warning("Cannot save to history: Database connection failed.")
        return False

    try:
        # Check if table exists
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute("""
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_NAME = 'sql_query_history' AND TABLE_SCHEMA = 'dbo'
                """)
                exists = result.fetchone()[0] > 0

                if exists:
                    connection.execute("""
                        INSERT INTO sql_query_history (natural_language_query, sql_query)
                        VALUES (?, ?)
                    """, (nl_query, sql_query))
        else:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_NAME = 'sql_query_history' AND TABLE_SCHEMA = 'dbo'
            """)
            exists = cursor.fetchone()[0] > 0

            if exists:
                cursor.execute("""
                    INSERT INTO sql_query_history (natural_language_query, sql_query)
                    VALUES (?, ?)
                """, (nl_query, sql_query))
                conn.commit()
            cursor.close()

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
        query = f"""
            SELECT TOP {limit} timestamp, natural_language_query, sql_query 
            FROM sql_query_history 
            ORDER BY timestamp DESC
        """

        if hasattr(conn, 'connect'):
            return pd.read_sql(query, conn)
        else:
            return pd.read_sql(query, conn)

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

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                for account_id in account_ids:
                    try:
                        connection.execute("""
                            INSERT INTO dormant_flags
                            (account_id, flag_instruction, flag_reason, flag_days, flagged_by, timestamp)
                            VALUES (?, ?, ?, ?, ?, GETDATE())
                        """, (
                            account_id,
                            flag_instruction,
                            f"Automated flagging via {flag_instruction}",
                            days_threshold,
                            "system"
                        ))
                        success_count += 1
                    except Exception:
                        continue
        else:
            cursor = conn.cursor()
            for account_id in account_ids:
                try:
                    cursor.execute("""
                        INSERT INTO dormant_flags
                        (account_id, flag_instruction, flag_reason, flag_days, flagged_by, timestamp)
                        VALUES (?, ?, ?, ?, ?, GETDATE())
                    """, (
                        account_id,
                        flag_instruction,
                        f"Automated flagging via {flag_instruction}",
                        days_threshold,
                        "system"
                    ))
                    success_count += 1
                except Exception:
                    continue

            conn.commit()
            cursor.close()

        if success_count == len(account_ids):
            return True, f"Successfully logged all {success_count} account flags."
        elif success_count > 0:
            return True, f"Partially successful: Logged {success_count} of {len(account_ids)} account flags."
        else:
            return False, "Failed to log any account flags."

    except Exception as e:
        return False, f"Error logging flags: {str(e)}"


def get_table_data(table_name, limit=1000):
    """Get data from a specific table."""
    conn = get_db_connection()
    if conn is None:
        return None

    try:
        query = f"SELECT TOP {limit} * FROM {table_name}"
        return pd.read_sql(query, conn)
    except Exception as e:
        st.error(f"Error retrieving data from {table_name}: {e}")
        return None


def execute_custom_query(query):
    """Execute a custom SQL query and return results."""
    conn = get_db_connection()
    if conn is None:
        st.error("No database connection available")
        return None

    try:
        result = pd.read_sql(query, conn)
        return result
    except Exception as e:
        st.error(f"Error executing query: {e}")
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