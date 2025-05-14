# data_transfer.py
import pandas as pd
import pyodbc
import sqlite3
import streamlit as st
import os
from datetime import datetime
from azure_connection import get_azure_connection


def get_sqlite_connection(db_path="app_data.db"):
    """Get connection to SQLite database for local app data storage."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        return sqlite3.connect(db_path)
    except Exception as e:
        st.error(f"Error connecting to SQLite database: {e}")
        return None


def extract_data_from_azure(query=None):
    """
    Extract data from Azure using SQL query.
    If no query is provided, get all accounts data.
    """
    conn = get_azure_connection()
    if not conn:
        st.error("Failed to connect to Azure database")
        return None

    try:
        # Default query if none provided
        if query is None:
            query = """
            SELECT 
                Account_ID, Customer_ID, Account_Type, Currency, Account_Creation_Date, 
                Current_Balance, Date_Last_Bank_Initiated_Activity, Date_Last_Customer_Communication_Any_Type, 
                FTD_Maturity_Date, FTD_Auto_Renewal, Date_Last_FTD_Renewal_Claim_Request, 
                Inv_Maturity_Redemption_Date, SDB_Charges_Outstanding, Date_SDB_Charges_Became_Outstanding, 
                SDB_Tenant_Communication_Received, Unclaimed_Item_Trigger_Date, Unclaimed_Item_Amount, 
                Date_Last_Cust_Initiated_Activity, Bank_Contact_Attempted_Post_Dormancy_Trigger, 
                Date_Last_Bank_Contact_Attempt, Customer_Responded_to_Bank_Contact, 
                Date_Claim_Received, Claim_Successful, Amount_Paid_on_Claim, 
                Scenario_Notes, Customer_Address_Known, Customer_Has_Active_Liability_Account, 
                Customer_Has_Litigation_Regulatory_Reqs, Holder_Has_Activity_On_Any_Other_Account, 
                Is_Asset_Only_Customer_Type, Expected_Account_Dormant, Expected_Requires_Article_3_Process, 
                Expected_Transfer_to_CB_Due, KYC_Status, Branch
            FROM accounts_data
            """

        # Read data into DataFrame
        df = pd.read_sql(query, conn)

        # Display extraction stats
        if df is not None and not df.empty:
            st.sidebar.success(f"✅ Extracted {len(df)} rows from Azure")
        else:
            st.sidebar.warning("⚠️ No data extracted from Azure")

        conn.close()
        return df
    except Exception as e:
        st.error(f"Error extracting data from Azure: {e}")
        if conn:
            conn.close()
        return None


def init_sqlite_db(conn=None):
    """Initialize SQLite database for the app to use with SQL Bot."""
    if conn is None:
        conn = get_sqlite_connection()
        if conn is None:
            return False

    try:
        cursor = conn.cursor()

        # Create accounts_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts_data (
            Account_ID TEXT,
            Customer_ID TEXT,
            Account_Type TEXT,
            Currency TEXT,
            Account_Creation_Date TEXT,
            Current_Balance REAL,
            Date_Last_Bank_Initiated_Activity TEXT,
            Date_Last_Customer_Communication_Any_Type TEXT,
            FTD_Maturity_Date TEXT,
            FTD_Auto_Renewal TEXT,
            Date_Last_FTD_Renewal_Claim_Request TEXT,
            Inv_Maturity_Redemption_Date TEXT,
            SDB_Charges_Outstanding REAL,
            Date_SDB_Charges_Became_Outstanding TEXT,
            SDB_Tenant_Communication_Received TEXT,
            Unclaimed_Item_Trigger_Date TEXT,
            Unclaimed_Item_Amount REAL,
            Date_Last_Cust_Initiated_Activity TEXT,
            Bank_Contact_Attempted_Post_Dormancy_Trigger TEXT,
            Date_Last_Bank_Contact_Attempt TEXT,
            Customer_Responded_to_Bank_Contact TEXT,
            Date_Claim_Received TEXT,
            Claim_Successful TEXT,
            Amount_Paid_on_Claim REAL,
            Scenario_Notes TEXT,
            Customer_Address_Known TEXT,
            Customer_Has_Active_Liability_Account TEXT,
            Customer_Has_Litigation_Regulatory_Reqs TEXT,
            Holder_Has_Activity_On_Any_Other_Account TEXT,
            Is_Asset_Only_Customer_Type TEXT,
            Expected_Account_Dormant TEXT,
            Expected_Requires_Article_3_Process TEXT,
            Expected_Transfer_to_CB_Due TEXT,
            KYC_Status TEXT,
            Branch TEXT,
            Email_Contact_Attempt TEXT,
            SMS_Contact_Attempt TEXT,
            Phone_Call_Attempt TEXT
        )
        """)

        # Create dormant_flags table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dormant_flags (
            account_id TEXT PRIMARY KEY,
            flag_instruction TEXT,
            timestamp TEXT
        )
        """)

        # Create dormant_ledger table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dormant_ledger (
            account_id TEXT PRIMARY KEY,
            classification TEXT,
            timestamp TEXT
        )
        """)

        # Create insight_log table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS insight_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            observation TEXT,
            trend TEXT,
            insight TEXT,
            action TEXT
        )
        """)

        # Create sql_query_history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sql_query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            natural_language_query TEXT,
            sql_query TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()
        st.sidebar.success("✅ Local SQLite database schema initialized")
        return True

    except Exception as e:
        st.error(f"Error initializing SQLite database: {e}")
        return False
    finally:
        if conn:
            conn.close()


def load_data_to_sqlite(df, table_name="accounts_data"):
    """Load DataFrame into SQLite database for local app use."""
    if df is None or df.empty:
        st.warning("No data to load into SQLite")
        return False

    conn = get_sqlite_connection()
    if conn is None:
        return False

    try:
        # Handle datetime columns
        for col in df.select_dtypes(include=['datetime64']).columns:
            df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else None)

        # Drop all existing data in the table
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")

        # Load new data
        df.to_sql(table_name, conn, if_exists='append', index=False)

        conn.commit()
        st.sidebar.success(f"✅ Successfully loaded {len(df)} rows into local SQLite database")
        return True
    except Exception as e:
        st.error(f"Error loading data into SQLite: {e}")
        return False
    finally:
        if conn:
            conn.close()


def transfer_data_from_azure_to_sqlite():
    """Transfer data from Azure to SQLite in a single function."""
    # Get data from Azure
    df = extract_data_from_azure()
    if df is None or df.empty:
        return False, "Failed to get data from Azure"

    # Initialize SQLite DB
    init_sqlite_db()

    # Load data to SQLite
    success = load_data_to_sqlite(df)

    if success:
        return True, f"Successfully transferred {len(df)} rows to local database"
    else:
        return False, "Failed to transfer data to local database"


def get_sqlite_schema():
    """Get the schema of the SQLite database for use with SQL Bot."""
    conn = get_sqlite_connection()
    if conn is None:
        return None

    try:
        cursor = conn.cursor()
        schema_info = {}

        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            # Get column info for each table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            column_details = []
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                nullable = "NULL" if col[3] == 0 else "NOT NULL"
                column_details.append((col_name, f"{col_type} {nullable}"))

            schema_info[table_name] = column_details

        return schema_info
    except Exception as e:
        st.error(f"Error retrieving SQLite schema: {e}")
        return None
    finally:
        if conn:
            conn.close()


def execute_sqlite_query(query):
    """Execute a query on the SQLite database."""
    conn = get_sqlite_connection()
    if conn is None:
        return None

    try:
        result = pd.read_sql_query(query, conn)
        return result
    except Exception as e:
        st.error(f"Error executing SQLite query: {e}")
        return None
    finally:
        if conn:
            conn.close()