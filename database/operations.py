import pyodbc
import pandas as pd
import streamlit as st
from datetime import datetime
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
        required_db_cols = ["Account_ID", "Account_Type", "Last_Transaction_Date", "Account_Status",
                            "Email_Contact_Attempt", "SMS_Contact_Attempt", "Phone_Call_Attempt", "KYC_Status",
                            "Branch"]
        cols_to_save = [col for col in required_db_cols if col in df.columns]
        if not cols_to_save:
            st.sidebar.error(f"No matching columns found in DataFrame for table '{table_name}'. Cannot save.")
            return False

        df_to_save = df[cols_to_save].copy()

        # Convert datetime columns to string format compatible with SQL Server DATETIME2
        for col in df_to_save.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
            # Convert NaT to None so pyodbc can handle it as SQL NULL
            df_to_save[col] = df_to_save[col].where(pd.notna(df_to_save[col]), None)
            # Format valid dates
            df_to_save[col] = df_to_save[col].apply(
                lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f') if isinstance(x, datetime) else None)

        # Convert other relevant columns to string to handle potential mixed types gracefully during insertion
        for col in ['Account_ID', 'Account_Type', 'Account_Status', 'Email_Contact_Attempt',
                    'SMS_Contact_Attempt', 'Phone_Call_Attempt', 'KYC_Status', 'Branch']:
            if col in df_to_save.columns:
                df_to_save[col] = df_to_save[col].astype(str)
                # Replace 'None' string resulting from NaT or actual None in conversion with 'Unknown' or empty string if appropriate
                # Using 'Unknown' as per parsing logic, but consider if empty string is better for DB NVARCHAR
                df_to_save[col] = df_to_save[col].replace('None', 'Unknown')

        with conn.cursor() as cursor:
            # Check if table exists before truncating
            cursor.execute(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", (table_name,))
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
                return False

            cursor.execute(f"TRUNCATE TABLE {table_name}")
            # conn.commit() # Commit truncate immediately

            # Prepare for bulk insert or row-by-row insert
            # Row-by-row is simpler but slower for large datasets. For demonstration, it's okay.
            # For production, consider using `executemany` or a dedicated bulk insert library.

            # Build the INSERT statement template
            placeholders = ','.join(['?'] * len(cols_to_save))
            columns_str = ','.join([f'[{c}]' for c in cols_to_save])  # Enclose column names in brackets for safety
            insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            # Prepare values as tuples for executemany
            # Handle potential None values for dates/nullable columns
            values_to_insert = []
            for index, row in df_to_save.iterrows():
                # Convert pandas NaT to Python None explicitly for pyodbc
                prepared_values = []
                for col in cols_to_save:
                    value = row[col]
                    if pd.isna(value):  # Check for pandas missing values (NaT for dates, NaN for numeric)
                        prepared_values.append(None)
                    else:
                        prepared_values.append(value)  # Keep other types as is (they should be strings/formatted dates)
                values_to_insert.append(tuple(prepared_values))

            if values_to_insert:
                # Use executemany for better performance than row-by-row execute
                cursor.executemany(insert_sql, values_to_insert)

            conn.commit()
        return True
    except pyodbc.Error as e:
        st.sidebar.error(f"Database Save Error ('{table_name}'): {e}. Check data compatibility or constraints.")
        return False
    except KeyError as e:
        st.sidebar.error(f"Missing expected column during save preparation: {e}")
        return False
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during DB save ('{table_name}'): {e}")
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
            cursor.execute(insert_sql, (timestamp, str(observation)[:4000], str(trend)[:4000], str(insight)[:4000],
                                        str(action)[
                                        :4000]))  # Truncate to fit MAX size if needed, or ensure column is NVARCHAR(MAX)
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