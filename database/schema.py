import pyodbc
import streamlit as st
from .connection import get_db_connection
from config import DB_NAME, DB_SERVER


def init_db():
    """ Initializes the Azure SQL database and tables using the default connection."""
    conn = get_db_connection()
    if conn is None:
        st.error("Cannot initialize database: Default DB connection failed.")
        return False  # Do not stop execution, allow app to run in disconnected mode

    try:
        with conn:
            cursor = conn.cursor()

            # Check if tables exist before creating them
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'accounts_data')
                CREATE TABLE accounts_data (
                    Account_ID NVARCHAR(255),
                    Customer_ID NVARCHAR(255),
                    Account_Type NVARCHAR(255),
                    Currency NVARCHAR(50),
                    Account_Creation_Date DATE,
                    Current_Balance DECIMAL(18, 2),
                    Date_Last_Bank_Initiated_Activity DATE,
                    Date_Last_Customer_Communication_Any_Type DATE,
                    FTD_Maturity_Date DATE,
                    FTD_Auto_Renewal NVARCHAR(10),
                    Date_Last_FTD_Renewal_Claim_Request DATE,
                    Inv_Maturity_Redemption_Date DATE,
                    SDB_Charges_Outstanding DECIMAL(18, 2),
                    Date_SDB_Charges_Became_Outstanding DATE,
                    SDB_Tenant_Communication_Received NVARCHAR(10),
                    Unclaimed_Item_Trigger_Date DATE,
                    Unclaimed_Item_Amount DECIMAL(18, 2),
                    Date_Last_Cust_Initiated_Activity DATE,
                    Bank_Contact_Attempted_Post_Dormancy_Trigger NVARCHAR(10),
                    Date_Last_Bank_Contact_Attempt DATE,
                    Customer_Responded_to_Bank_Contact NVARCHAR(10),
                    Date_Claim_Received DATE,
                    Claim_Successful NVARCHAR(10),
                    Amount_Paid_on_Claim DECIMAL(18, 2),
                    Scenario_Notes NVARCHAR(MAX),
                    Customer_Address_Known NVARCHAR(10),
                    Customer_Has_Active_Liability_Account NVARCHAR(10),
                    Customer_Has_Litigation_Regulatory_Reqs NVARCHAR(10),
                    Holder_Has_Activity_On_Any_Other_Account NVARCHAR(10),
                    Is_Asset_Only_Customer_Type NVARCHAR(10),
                    Expected_Account_Dormant NVARCHAR(10),
                    Expected_Requires_Article_3_Process NVARCHAR(10),
                    Expected_Transfer_to_CB_Due NVARCHAR(10)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags')
                CREATE TABLE dormant_flags (
                    account_id NVARCHAR(255) PRIMARY KEY,
                    flag_instruction NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_ledger')
                CREATE TABLE dormant_ledger (
                    account_id NVARCHAR(255) PRIMARY KEY,
                    classification NVARCHAR(255),
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'insight_log')
                CREATE TABLE insight_log (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP,
                    observation NVARCHAR(MAX),
                    trend NVARCHAR(MAX),
                    insight NVARCHAR(MAX),
                    action NVARCHAR(MAX)
                )
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'sql_query_history')
                CREATE TABLE sql_query_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    natural_language_query NVARCHAR(MAX),
                    sql_query NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes using the correct syntax for Azure SQL
            # Check if index exists before creating it
            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_accounts_data_Account_ID' AND object_id = OBJECT_ID('accounts_data'))
                BEGIN
                    CREATE INDEX IX_accounts_data_Account_ID ON accounts_data (Account_ID);
                END
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_accounts_data_Customer_ID' AND object_id = OBJECT_ID('accounts_data'))
                BEGIN
                    CREATE INDEX IX_accounts_data_Customer_ID ON accounts_data (Customer_ID);
                END
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_accounts_data_Account_Type' AND object_id = OBJECT_ID('accounts_data'))
                BEGIN
                    CREATE INDEX IX_accounts_data_Account_Type ON accounts_data (Account_Type);
                END
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_accounts_data_Date_Last_Cust_Initiated_Activity' 
                               AND object_id = OBJECT_ID('accounts_data'))
                BEGIN
                    CREATE INDEX IX_accounts_data_Date_Last_Cust_Initiated_Activity 
                    ON accounts_data (Date_Last_Cust_Initiated_Activity);
                END
            """)

            cursor.execute("""
                IF NOT EXISTS (SELECT * FROM sys.indexes WHERE name = 'IX_accounts_data_Expected_Account_Dormant' 
                               AND object_id = OBJECT_ID('accounts_data'))
                BEGIN
                    CREATE INDEX IX_accounts_data_Expected_Account_Dormant
                    ON accounts_data (Expected_Account_Dormant);
                END
            """)

            conn.commit()
            st.sidebar.success("✅ Database schema initialized/verified.")
        return True
    except pyodbc.Error as e:
        st.error(f"Database Initialization Error: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during DB initialization: {e}")
        return False


@st.cache_data(show_spinner="Fetching database schema for SQL Bot...", ttl="1h")  # Cache schema for 1 hour
def get_db_schema():
    """Fetches schema for the default database."""
    conn = get_db_connection()  # Always use the default cached connection helper

    if conn is None:
        # Error message is already shown by get_db_connection
        return None

    schema_info = {}
    db_identifier = f"default database '{DB_NAME}' on '{DB_SERVER}'"

    try:
        with conn:  # Use context manager with the cached connection
            cursor = conn.cursor()
            # Fetch tables from the default database
            cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
            tables = cursor.fetchall()
            for table_row in tables:
                table_name = table_row[0]
                # Fetch columns for each table
                cursor.execute(
                    f"SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?",
                    (table_name,))
                columns_raw = cursor.fetchall()
                column_details = []
                for col_raw in columns_raw:
                    col_name, data_type, max_length, is_nullable = col_raw[0], col_raw[1], col_raw[2], col_raw[3]
                    type_info = f"{data_type}"
                    if data_type in ('varchar', 'nvarchar', 'char', 'nchar', 'binary', 'varbinary'):
                        if max_length == -1:
                            type_info += "(MAX)"
                        elif max_length is not None:
                            type_info += f"({max_length})"
                    elif data_type in ('decimal', 'numeric'):
                        # Fetch precision and scale
                        try:
                            cursor.execute(
                                f"SELECT NUMERIC_PRECISION, NUMERIC_SCALE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ? AND COLUMN_NAME = ?",
                                (table_name, col_name))
                            prec_scale = cursor.fetchone()
                            if prec_scale:
                                precision, scale = prec_scale
                                type_info += f"({precision},{scale})"
                        except Exception:
                            pass  # Ignore if fetching precision/scale fails
                    elif data_type in ('float', 'real'):
                        # Fetch precision for float/real
                        try:
                            cursor.execute(
                                f"SELECT NUMERIC_PRECISION FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ? AND COLUMN_NAME = ?",
                                (table_name, col_name))
                            precision = cursor.fetchone()
                            if precision and precision[0] is not None:
                                type_info += f"({precision[0]})"
                        except Exception:
                            pass  # Ignore

                    nullable_status = "NULL" if is_nullable == "YES" else "NOT NULL"

                    column_details.append(
                        (col_name, f"{type_info} {nullable_status}"))  # Add nullable status to schema info

                schema_info[table_name] = column_details

        st.sidebar.success("✅ Database schema fetched.")
        return schema_info

    except pyodbc.Error as e:
        st.error(f"SQL Bot: Database error fetching schema from {db_identifier}: {e}")
        return None
    except Exception as e:
        st.error(f"SQL Bot: Unexpected error fetching schema from {db_identifier}: {e}")
        return None