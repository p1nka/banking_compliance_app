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
        # Determine which columns from the DataFrame to save based on the table schema
        cursor = conn.cursor()
        cursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'")
        db_columns = [row[0] for row in cursor.fetchall()]

        # Find matching columns between DataFrame and DB table
        cols_to_save = [col for col in df.columns if col in db_columns]

        if not cols_to_save:
            st.sidebar.error(f"No matching columns found in DataFrame for table '{table_name}'. Cannot save.")
            return False

        # Prepare DataFrame for saving
        df_to_save = df[cols_to_save].copy()

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

        for col in boolean_like_columns:
            if col in df_to_save.columns:
                # Ensure values are standardized as 'Yes'/'No'/'Unknown'
                df_to_save[col] = df_to_save[col].astype(str)
                df_to_save[col] = df_to_save[col].replace({
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
            # Check if table exists before truncating
            cursor.execute(f"SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?", (table_name,))
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                st.sidebar.error(f"Table '{table_name}' does not exist in the database. Cannot save.")
                return False

            cursor.execute(f"TRUNCATE TABLE {table_name}")

            # Build the INSERT statement template
            placeholders = ','.join(['?'] * len(cols_to_save))
            columns_str = ','.join([f'[{c}]' for c in cols_to_save])  # Enclose column names in brackets
            insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"

            # Prepare values as tuples for executemany
            values_to_insert = []
            for index, row in df_to_save.iterrows():
                # Convert pandas NaN/NaT to Python None for pyodbc
                prepared_values = []
                for col in cols_to_save:
                    value = row[col]
                    if pd.isna(value):  # Check for pandas missing values (NaT, NaN)
                        prepared_values.append(None)
                    else:
                        prepared_values.append(value)
                values_to_insert.append(tuple(prepared_values))

            if values_to_insert:
                # Use executemany for better performance
                cursor.executemany(insert_sql, values_to_insert)

            conn.commit()
        return True
    except pyodbc.Error as e:
        st.sidebar.error(f"Database Save Error ('{table_name}'): {e}. Check data compatibility or constraints.")
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