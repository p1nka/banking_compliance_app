# --- START OF FILE transfer.py ---

import pandas as pd
import sqlite3
import streamlit as st
import os
from datetime import datetime
# FIX: Import from the correct location
from database.connection import get_db_connection


def get_sqlite_connection(db_path="app_data.db"):
    """Get connection to SQLite database for local app data storage."""
    try:
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        return sqlite3.connect(db_path)
    except Exception as e:
        st.error(f"Error connecting to SQLite database: {e}")
        return None


def extract_data_from_azure(query=None):
    """Extract data from Azure using SQL query."""
    # FIX: Use the correct connection function name
    conn = get_db_connection()
    if not conn:
        st.error("Failed to connect to Azure database")
        return None

    try:
        if query is None:
            # A more generic query to pull all columns from the main table
            query = "SELECT * FROM accounts_data"

        df = pd.read_sql(query, conn)
        if df is not None and not df.empty:
            st.sidebar.success(f"✅ Extracted {len(df)} rows from Azure")
        else:
            st.sidebar.warning("⚠️ No data extracted from Azure")

        return df
    except Exception as e:
        st.error(f"Error extracting data from Azure: {e}")
        return None
    finally:
        # The engine manages its own connections, but good practice to dispose
        if conn:
            conn.dispose()


def init_sqlite_db(conn=None):
    """
    Initialize SQLite database schema.
    NOTE: This schema should be kept in sync with the one in `database/schema.py`.
    """
    close_conn = False
    if conn is None:
        conn = get_sqlite_connection()
        if conn is None: return False
        close_conn = True

    try:
        cursor = conn.cursor()
        # Use the same table structure as in schema.py for consistency
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts_data (
            Account_ID TEXT PRIMARY KEY, Customer_ID TEXT, Account_Type TEXT, Currency TEXT,
            Account_Creation_Date TEXT, Current_Balance REAL, Date_Last_Bank_Initiated_Activity TEXT,
            Date_Last_Customer_Communication_Any_Type TEXT, FTD_Maturity_Date TEXT, FTD_Auto_Renewal TEXT,
            Date_Last_FTD_Renewal_Claim_Request TEXT, Inv_Maturity_Redemption_Date TEXT,
            SDB_Charges_Outstanding TEXT, Date_SDB_Charges_Became_Outstanding TEXT,
            SDB_Tenant_Communication_Received TEXT, Unclaimed_Item_Trigger_Date TEXT,
            Unclaimed_Item_Amount REAL, Date_Last_Cust_Initiated_Activity TEXT,
            Bank_Contact_Attempted_Post_Dormancy_Trigger TEXT, Date_Last_Bank_Contact_Attempt TEXT,
            Customer_Responded_to_Bank_Contact TEXT, Date_Claim_Received TEXT, Claim_Successful TEXT,
            Amount_Paid_on_Claim REAL, Scenario_Notes TEXT, Customer_Address_Known TEXT,
            Customer_Has_Active_Liability_Account TEXT, Customer_Has_Litigation_Regulatory_Reqs TEXT,
            Holder_Has_Activity_On_Any_Other_Account TEXT, Is_Asset_Only_Customer_Type TEXT,
            Expected_Account_Dormant TEXT, Expected_Requires_Article_3_Process TEXT,
            Expected_Transfer_to_CB_Due TEXT
        )""")
        conn.commit()
        st.sidebar.success("✅ Local SQLite database schema initialized")
        return True
    except Exception as e:
        st.error(f"Error initializing SQLite database: {e}")
        return False
    finally:
        if conn and close_conn:
            conn.close()


def load_data_to_sqlite(df, table_name="accounts_data"):
    """Load DataFrame into SQLite database with transactional safety."""
    if df is None or df.empty:
        st.warning("No data to load into SQLite")
        return False

    conn = get_sqlite_connection()
    if conn is None: return False

    try:
        # Convert datetime columns to string format for SQLite compatibility
        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # FIX: Use a transaction to ensure atomicity
        conn.execute(f"DELETE FROM {table_name}")
        df.to_sql(table_name, conn, if_exists='append', index=False)
        conn.commit()
        st.sidebar.success(f"✅ Loaded {len(df)} rows into local SQLite database")
        return True
    except Exception as e:
        st.error(f"Error loading data into SQLite: {e}")
        conn.rollback()  # Rollback changes on error
        return False
    finally:
        if conn:
            conn.close()


def transfer_data_from_azure_to_sqlite():
    """Transfer data from Azure to SQLite in a single function."""
    df = extract_data_from_azure()
    if df is None or df.empty:
        return False, "Failed to get data from Azure"

    init_sqlite_db()
    success = load_data_to_sqlite(df)

    if success:
        return True, f"Successfully transferred {len(df)} rows to local database"
    else:
        return False, "Failed to transfer data to local database"