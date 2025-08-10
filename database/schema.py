# database/schema.py
"""
Database schema management for Azure SQL Database.
Enhanced with better error handling and Azure SQL compatibility.
"""

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback
import pandas as pd
from database.connection import get_db_connection


def init_db():
    """
    Initialize the Azure SQL database and tables.
    Enhanced with better error handling and connection management.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            st.sidebar.warning("⚠️ Database connection failed. App will run in offline mode.")
            return False

        # Test if connection is working
        try:
            success, message = test_connection(conn)
            if success:
                st.sidebar.success("✅ Database connection verified")
                st.sidebar.info(f"📊 {message}")
            else:
                st.sidebar.error(f"⚠️ Database connection test failed: {message}")
                return False
        except Exception as test_error:
            st.sidebar.error(f"⚠️ Database connection test failed: {test_error}")
            return False

        # Initialize tables
        success = create_tables_if_not_exist(conn)

        if success:
            st.sidebar.success("✅ Database schema verified/initialized")
        else:
            st.sidebar.warning("⚠️ Database schema initialization had issues")

        return success

    except Exception as e:
        st.sidebar.error(f"❌ Database initialization error: {str(e)}")
        st.sidebar.info("App will continue in offline mode")
        return False


def test_connection(conn):
    """
    Test the database connection with Azure SQL specific queries.
    Returns (success: bool, message: str)
    """
    try:
        if hasattr(conn, 'connect'):
            # SQLAlchemy engine
            with conn.connect() as test_conn:
                result = test_conn.execute(
                    "SELECT GETDATE() as current_time, DB_NAME() as database_name, @@VERSION as version")
                row = result.fetchone()
                return True, f"Connected to '{row[1]}' at {row[0]}"
        elif hasattr(conn, 'execute'):
            # SQLAlchemy connection
            result = conn.execute("SELECT GETDATE() as current_time, DB_NAME() as database_name")
            row = result.fetchone()
            return True, f"Connected to '{row[1]}' at {row[0]}"
        else:
            # Direct connection (pyodbc/pymssql)
            cursor = conn.cursor()
            cursor.execute("SELECT GETDATE(), DB_NAME()")
            row = cursor.fetchone()
            cursor.close()
            return True, f"Connected to '{row[1]}' at {row[0]}"
    except Exception as e:
        return False, str(e)


def create_tables_if_not_exist(conn):
    """
    Create necessary tables if they don't exist.
    Works with both SQLAlchemy engines and direct connections.
    """
    try:
        if hasattr(conn, 'connect'):
            # SQLAlchemy engine - use connection context
            return create_tables_sqlalchemy_engine(conn)
        elif hasattr(conn, 'execute'):
            # SQLAlchemy connection
            return create_tables_sqlalchemy_connection(conn)
        else:
            # Direct connection (pymssql)
            return create_tables_direct(conn)
    except Exception as e:
        st.sidebar.error(f"Table creation error: {e}")
        return False


def create_tables_sqlalchemy_engine(engine):
    """Create tables using SQLAlchemy engine."""
    try:
        with engine.connect() as conn:
            # Use transaction for all table creation
            trans = conn.begin()
            try:
                # Check and create main accounts table
                if not table_exists(conn, 'accounts_data'):
                    st.sidebar.info("Creating accounts_data table...")
                    create_accounts_table_sqlalchemy_conn(conn)

                # Create support tables
                create_support_tables_sqlalchemy_conn(conn)

                trans.commit()
                return True
            except Exception as e:
                trans.rollback()
                raise e
    except Exception as e:
        st.sidebar.error(f"SQLAlchemy engine table creation failed: {e}")
        return False


def create_tables_sqlalchemy_connection(conn):
    """Create tables using SQLAlchemy connection."""
    try:
        # Check and create main accounts table
        if not table_exists(conn, 'accounts_data'):
            st.sidebar.info("Creating accounts_data table...")
            create_accounts_table_sqlalchemy_conn(conn)

        # Create support tables
        create_support_tables_sqlalchemy_conn(conn)

        return True
    except Exception as e:
        st.sidebar.error(f"SQLAlchemy connection table creation failed: {e}")
        return False


def create_tables_direct(conn):
    """Create tables using direct connection."""
    try:
        cursor = conn.cursor()

        # Check and create main accounts table
        if not table_exists_direct(cursor, 'accounts_data'):
            st.sidebar.info("Creating accounts_data table...")
            create_accounts_table_direct(cursor)

        # Create support tables
        create_support_tables_direct(cursor)

        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        st.sidebar.error(f"Direct table creation failed: {e}")
        return False


def table_exists(conn, table_name):
    """Check if table exists using SQLAlchemy connection."""
    try:
        result = conn.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = 'dbo'
        """, (table_name,))
        return result.fetchone()[0] > 0
    except:
        return False


def get_accounts_create_sql():
    """
    Returns the SQL for creating the comprehensive accounts_data table.
    This centralized function ensures schema consistency.
    """
    return """
    CREATE TABLE accounts_data (
        Account_ID NVARCHAR(255) NOT NULL PRIMARY KEY,
        Customer_ID NVARCHAR(255),
        Account_Type NVARCHAR(255),
        Currency NVARCHAR(50),
        Account_Creation_Date DATE,
        Current_Balance DECIMAL(18, 2),
        Date_Last_Bank_Initiated_Activity DATE,
        Date_Last_Customer_Communication_Any_Type DATE,
        FTD_Maturity_Date DATE,
        FTD_Auto_Renewal NVARCHAR(50),
        Date_Last_FTD_Renewal_Claim_Request DATE,
        Inv_Maturity_Redemption_Date DATE,
        SDB_Charges_Outstanding NVARCHAR(50),
        Date_SDB_Charges_Became_Outstanding DATE,
        SDB_Tenant_Communication_Received NVARCHAR(50),
        Unclaimed_Item_Trigger_Date DATE,
        Unclaimed_Item_Amount DECIMAL(18, 2),
        Date_Last_Cust_Initiated_Activity DATE,
        Bank_Contact_Attempted_Post_Dormancy_Trigger NVARCHAR(50),
        Date_Last_Bank_Contact_Attempt DATE,
        Customer_Responded_to_Bank_Contact NVARCHAR(50),
        Date_Claim_Received DATE,
        Claim_Successful NVARCHAR(50),
        Amount_Paid_on_Claim DECIMAL(18, 2),
        Scenario_Notes NVARCHAR(MAX),
        Customer_Address_Known NVARCHAR(50),
        Customer_Has_Active_Liability_Account NVARCHAR(50),
        Customer_Has_Litigation_Regulatory_Reqs NVARCHAR(50),
        Holder_Has_Activity_On_Any_Other_Account NVARCHAR(50),
        Is_Asset_Only_Customer_Type NVARCHAR(50),
        Expected_Account_Dormant NVARCHAR(50),
        Expected_Requires_Article_3_Process NVARCHAR(50),
        Expected_Transfer_to_CB_Due NVARCHAR(50)
    )
    """


def create_accounts_table_sqlalchemy_conn(conn):
    """Create accounts_data table using SQLAlchemy connection."""
    create_sql = get_accounts_create_sql()
    conn.execute(create_sql)


def create_accounts_table_direct(cursor):
    """Create accounts_data table using direct cursor."""
    create_sql = get_accounts_create_sql()
    cursor.execute(create_sql)


def create_support_tables_sqlalchemy_conn(conn):
    """Create support tables using SQLAlchemy connection."""
    # Create dormant_flags table
    try:
        if not table_exists(conn, 'dormant_flags'):
            dormant_flags_sql = """
            CREATE TABLE dormant_flags (
                id INT IDENTITY(1,1) PRIMARY KEY,
                account_id NVARCHAR(255) NOT NULL,
                flag_instruction NVARCHAR(MAX) NOT NULL,
                flag_reason NVARCHAR(MAX),
                flag_days INT,
                flagged_by NVARCHAR(255) DEFAULT 'system',
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(dormant_flags_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_flags table: {e}")

    # Create dormant_ledger table
    try:
        if not table_exists(conn, 'dormant_ledger'):
            dormant_ledger_sql = """
            CREATE TABLE dormant_ledger (
                id INT IDENTITY(1,1) PRIMARY KEY,
                account_id NVARCHAR(255) NOT NULL,
                classification NVARCHAR(255) NOT NULL,
                classification_reason NVARCHAR(MAX),
                classified_by NVARCHAR(255) DEFAULT 'system',
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(dormant_ledger_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_ledger table: {e}")

    # Create insight_log table
    try:
        if not table_exists(conn, 'insight_log'):
            insight_log_sql = """
            CREATE TABLE insight_log (
                id INT IDENTITY(1,1) PRIMARY KEY,
                timestamp DATETIME2 DEFAULT GETDATE(),
                observation NVARCHAR(MAX),
                trend NVARCHAR(MAX),
                insight NVARCHAR(MAX),
                action NVARCHAR(MAX)
            )
            """
            conn.execute(insight_log_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create insight_log table: {e}")

    # Create sql_query_history table
    try:
        if not table_exists(conn, 'sql_query_history'):
            history_sql = """
            CREATE TABLE sql_query_history (
                id INT IDENTITY(1,1) PRIMARY KEY,
                natural_language_query NVARCHAR(MAX),
                sql_query NVARCHAR(MAX),
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(history_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create sql_query_history table: {e}")

    # Create analysis_results table
    try:
        if not table_exists(conn, 'analysis_results'):
            analysis_sql = """
            CREATE TABLE analysis_results (
                id INT IDENTITY(1,1) PRIMARY KEY,
                analysis_type NVARCHAR(100) NOT NULL,
                analysis_name NVARCHAR(255) NOT NULL,
                result_data NVARCHAR(MAX) NOT NULL,
                result_summary NVARCHAR(MAX),
                record_count INT,
                created_by NVARCHAR(100) DEFAULT 'system',
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(analysis_sql)

            # Create index for better performance
            index_sql = """
            CREATE INDEX IX_analysis_results_type_name 
            ON analysis_results (analysis_type, analysis_name)
            """
            conn.execute(index_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create analysis_results table: {e}")


def create_support_tables_direct(cursor):
    """Create support tables using direct cursor."""
    # Create dormant_flags table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'dormant_flags' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE dormant_flags (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    account_id NVARCHAR(255) NOT NULL,
                    flag_instruction NVARCHAR(MAX) NOT NULL,
                    flag_reason NVARCHAR(MAX),
                    flag_days INT,
                    flagged_by NVARCHAR(255) DEFAULT 'system',
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_flags table: {e}")

    # Create dormant_ledger table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'dormant_ledger' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE dormant_ledger (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    account_id NVARCHAR(255) NOT NULL,
                    classification NVARCHAR(255) NOT NULL,
                    classification_reason NVARCHAR(MAX),
                    classified_by NVARCHAR(255) DEFAULT 'system',
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_ledger table: {e}")

    # Create insight_log table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'insight_log' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE insight_log (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    observation NVARCHAR(MAX),
                    trend NVARCHAR(MAX),
                    insight NVARCHAR(MAX),
                    action NVARCHAR(MAX)
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create insight_log table: {e}")

    # Create sql_query_history table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'sql_query_history' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE sql_query_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    natural_language_query NVARCHAR(MAX),
                    sql_query NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create sql_query_history table: {e}")

    # Create analysis_results table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'analysis_results' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE analysis_results (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    analysis_type NVARCHAR(100) NOT NULL,
                    analysis_name NVARCHAR(255) NOT NULL,
                    result_data NVARCHAR(MAX) NOT NULL,
                    result_summary NVARCHAR(MAX),
                    record_count INT,
                    created_by NVARCHAR(100) DEFAULT 'system',
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)

        # Create index if it doesn't exist
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sys.indexes 
                          WHERE name = 'IX_analysis_results_type_name' 
                          AND object_id = OBJECT_ID('analysis_results'))
            BEGIN
                CREATE INDEX IX_analysis_results_type_name 
                ON analysis_results (analysis_type, analysis_name)
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create analysis_results table: {e}")


@st.cache_data(ttl=600)
def get_db_schema():
    """
    MAIN SCHEMA FUNCTION - Enhanced with datetime detection and multiple fallback methods.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            st.sidebar.warning("Cannot fetch schema: No database connection")
            return get_fallback_schema()

        try:
            enhanced_schema = enhanced_get_schema_with_datetime_info(conn)
            if enhanced_schema:
                st.sidebar.info(f"✅ Enhanced schema loaded: {len(enhanced_schema)} tables")
                return enhanced_schema
        except Exception as enhanced_error:
            st.sidebar.warning(f"Enhanced schema failed: {str(enhanced_error)[:100]}")

        # Fallback to original method if enhanced fails
        try:
            schema_info = get_schema_sys_tables(conn)
            if schema_info:
                st.sidebar.info(f"✅ Schema loaded via sys.tables: {len(schema_info)} tables")
                return schema_info
        except Exception as method_error:
            st.sidebar.warning(f"Schema method 'sys.tables' failed: {str(method_error)[:100]}")

        st.sidebar.warning("⚠️ All schema methods failed, using fallback schema")
        return get_fallback_schema()

    except Exception as e:
        st.sidebar.error(f"Schema fetch error: {e}")
        return get_fallback_schema()


def enhanced_get_schema_with_datetime_info(conn):
    """
    Enhanced schema detection that specifically handles datetime types properly.
    """
    try:
        enhanced_schema_query = """
        SELECT 
            t.name AS table_name,
            c.name AS column_name,
            ty.name AS data_type,
            c.max_length,
            c.precision,
            c.scale,
            c.is_nullable,
            CASE 
                WHEN ty.name IN ('datetime', 'datetime2', 'date', 'time', 'datetimeoffset') 
                THEN 'DATETIME_TYPE'
                WHEN ty.name IN ('int', 'bigint', 'decimal', 'float', 'money', 'numeric', 'real', 'smallint', 'smallmoney', 'tinyint')
                THEN 'NUMERIC_TYPE'
                ELSE 'OTHER_TYPE'
            END AS type_category,
            CASE 
                WHEN ty.name = 'datetime2' THEN CONCAT('datetime2(', c.scale, ')')
                WHEN ty.name = 'decimal' THEN CONCAT('decimal(', c.precision, ',', c.scale, ')')
                WHEN ty.name = 'nvarchar' AND c.max_length = -1 THEN 'nvarchar(MAX)'
                WHEN ty.name = 'nvarchar' THEN CONCAT('nvarchar(', c.max_length/2, ')')
                WHEN ty.name = 'varchar' AND c.max_length = -1 THEN 'varchar(MAX)'
                WHEN ty.name = 'varchar' THEN CONCAT('varchar(', c.max_length, ')')
                ELSE ty.name
            END AS formatted_type
        FROM sys.tables t
        INNER JOIN sys.columns c ON t.object_id = c.object_id
        INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id
        WHERE t.type = 'U'
        ORDER BY t.name, c.column_id
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(enhanced_schema_query)
                return _format_enhanced_schema_result(result)
        elif hasattr(conn, 'execute'):
            result = conn.execute(enhanced_schema_query)
            return _format_enhanced_schema_result(result)
        else:
            cursor = conn.cursor()
            cursor.execute(enhanced_schema_query)
            rows = cursor.fetchall()
            cursor.close()
            return _format_enhanced_schema_result_direct(rows)

    except Exception as e:
        st.error(f"Enhanced schema detection failed: {e}")
        return None


def _format_enhanced_schema_result(result):
    """Format enhanced schema result with datetime type information."""
    schema_info = {}
    datetime_columns = {}

    for row in result:
        table_name, column_name, data_type, max_length, precision, scale, is_nullable, type_category, formatted_type = row

        if table_name not in schema_info:
            schema_info[table_name] = []
            datetime_columns[table_name] = []

        nullable_info = "NULL" if is_nullable else "NOT NULL"
        schema_info[table_name].append((column_name, f"{formatted_type} {nullable_info}"))

        if type_category == 'DATETIME_TYPE':
            datetime_columns[table_name].append({
                'column': column_name, 'type': data_type, 'formatted_type': formatted_type,
                'precision': precision, 'scale': scale
            })

    if 'datetime_columns_info' not in st.session_state:
        st.session_state['datetime_columns_info'] = datetime_columns
    return schema_info


def _format_enhanced_schema_result_direct(rows):
    """Format enhanced schema result for direct connections."""
    schema_info = {}
    datetime_columns = {}

    for row in rows:
        table_name, column_name, data_type, max_length, precision, scale, is_nullable, type_category, formatted_type = row

        if table_name not in schema_info:
            schema_info[table_name] = []
            datetime_columns[table_name] = []

        nullable_info = "NULL" if is_nullable else "NOT NULL"
        schema_info[table_name].append((column_name, f"{formatted_type} {nullable_info}"))

        if type_category == 'DATETIME_TYPE':
            datetime_columns[table_name].append({
                'column': column_name, 'type': data_type, 'formatted_type': formatted_type,
                'precision': precision, 'scale': scale
            })

    st.session_state['datetime_columns_info'] = datetime_columns
    return schema_info


def get_schema_sys_tables(conn):
    """Get schema using sys.tables and sys.columns."""
    try:
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                return _execute_sys_tables_query(connection)
        else:
            return _execute_sys_tables_query_direct(conn)
    except Exception as e:
        raise Exception(f"sys.tables query failed: {e}")


def _execute_sys_tables_query(conn):
    """Execute sys.tables query for SQLAlchemy connections"""
    schema_query = """
    SELECT t.name AS table_name, c.name AS column_name, ty.name AS data_type, 
           c.max_length, c.is_nullable 
    FROM sys.tables t 
    INNER JOIN sys.columns c ON t.object_id = c.object_id 
    INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id 
    WHERE t.type = 'U' 
    ORDER BY t.name, c.column_id
    """
    result = conn.execute(schema_query)
    return _format_schema_from_result(result)


def _execute_sys_tables_query_direct(conn):
    """Execute sys.tables query for direct connections"""
    cursor = conn.cursor()
    cursor.execute("""
    SELECT t.name AS table_name, c.name AS column_name, ty.name AS data_type, 
           c.max_length, c.is_nullable 
    FROM sys.tables t 
    INNER JOIN sys.columns c ON t.object_id = c.object_id 
    INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id 
    WHERE t.type = 'U' 
    ORDER BY t.name, c.column_id
    """)
    rows = cursor.fetchall()
    cursor.close()
    return _format_schema_from_result(rows)


def _format_schema_from_result(result_rows):
    """Format schema query result into standard format"""
    schema_info = {}
    for row in result_rows:
        table_name, column_name, data_type, max_length, is_nullable = row
        if table_name not in schema_info:
            schema_info[table_name] = []

        if data_type in ('nvarchar', 'varchar') and max_length and max_length > 0:
            length = int(max_length / 2) if data_type == 'nvarchar' else max_length
            formatted_type = f"{data_type}({length if length > 0 else 'MAX'})"
        else:
            formatted_type = data_type
        nullable_info = "NULL" if is_nullable else "NOT NULL"
        schema_info[table_name].append((column_name, f"{formatted_type} {nullable_info}"))
    return schema_info


def get_fallback_schema():
    """Fallback schema when database schema cannot be retrieved"""
    return {
        'accounts_data': [
            ('Account_ID', 'NVARCHAR(255) NOT NULL'),
            ('Customer_ID', 'NVARCHAR(255)'),
            ('Account_Type', 'NVARCHAR(255)'),
            ('Currency', 'NVARCHAR(50)'),
            ('Account_Creation_Date', 'DATE'),
            ('Current_Balance', 'DECIMAL(18, 2)'),
            ('Date_Last_Bank_Initiated_Activity', 'DATE'),
            ('Date_Last_Customer_Communication_Any_Type', 'DATE'),
            ('FTD_Maturity_Date', 'DATE'),
            ('FTD_Auto_Renewal', 'NVARCHAR(50)'),
            ('Date_Last_FTD_Renewal_Claim_Request', 'DATE'),
            ('Inv_Maturity_Redemption_Date', 'DATE'),
            ('SDB_Charges_Outstanding', 'NVARCHAR(50)'),
            ('Date_SDB_Charges_Became_Outstanding', 'DATE'),
            ('SDB_Tenant_Communication_Received', 'NVARCHAR(50)'),
            ('Unclaimed_Item_Trigger_Date', 'DATE'),
            ('Unclaimed_Item_Amount', 'DECIMAL(18, 2)'),
            ('Date_Last_Cust_Initiated_Activity', 'DATE'),
            ('Bank_Contact_Attempted_Post_Dormancy_Trigger', 'NVARCHAR(50)'),
            ('Date_Last_Bank_Contact_Attempt', 'DATE'),
            ('Customer_Responded_to_Bank_Contact', 'NVARCHAR(50)'),
            ('Date_Claim_Received', 'DATE'),
            ('Claim_Successful', 'NVARCHAR(50)'),
            ('Amount_Paid_on_Claim', 'DECIMAL(18, 2)'),
            ('Scenario_Notes', 'NVARCHAR(MAX)'),
            ('Customer_Address_Known', 'NVARCHAR(50)'),
            ('Customer_Has_Active_Liability_Account', 'NVARCHAR(50)'),
            ('Customer_Has_Litigation_Regulatory_Reqs', 'NVARCHAR(50)'),
            ('Holder_Has_Activity_On_Any_Other_Account', 'NVARCHAR(50)'),
            ('Is_Asset_Only_Customer_Type', 'NVARCHAR(50)'),
            ('Expected_Account_Dormant', 'NVARCHAR(50)'),
            ('Expected_Requires_Article_3_Process', 'NVARCHAR(50)'),
            ('Expected_Transfer_to_CB_Due', 'NVARCHAR(50)')
        ],
        'dormant_flags': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('account_id', 'NVARCHAR(255) NOT NULL'),
            ('flag_instruction', 'NVARCHAR(MAX) NOT NULL'),
            ('flag_reason', 'NVARCHAR(MAX)'),
            ('flag_days', 'INT'),
            ('flagged_by', 'NVARCHAR(255)'),
            ('timestamp', 'DATETIME2 NULL')
        ],
        'dormant_ledger': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('account_id', 'NVARCHAR(255) NOT NULL'),
            ('classification', 'NVARCHAR(255) NOT NULL'),
            ('classification_reason', 'NVARCHAR(MAX)'),
            ('classified_by', 'NVARCHAR(255)'),
            ('timestamp', 'DATETIME2 NULL')
        ],
        'insight_log': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('timestamp', 'DATETIME2'),
            ('observation', 'NVARCHAR(MAX)'),
            ('trend', 'NVARCHAR(MAX)'),
            ('insight', 'NVARCHAR(MAX)'),
            ('action', 'NVARCHAR(MAX)')
        ],
        'sql_query_history': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('natural_language_query', 'NVARCHAR(MAX)'),
            ('sql_query', 'NVARCHAR(MAX)'),
            ('timestamp', 'DATETIME2')
        ],
        'analysis_results': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('analysis_type', 'NVARCHAR(100) NOT NULL'),
            ('analysis_name', 'NVARCHAR(255) NOT NULL'),
            ('result_data', 'NVARCHAR(MAX) NOT NULL'),
            ('result_summary', 'NVARCHAR(MAX)'),
            ('record_count', 'INT'),
            ('created_by', 'NVARCHAR(100)'),
            ('timestamp', 'DATETIME2')
        ]
    }


def get_datetime_columns_info():
    """
    Get information about datetime columns from session state.
    """
    return st.session_state.get('datetime_columns_info', {})


def is_datetime_column_in_table(table_name, column_name):
    """
    Check if a specific column in a table is a datetime column.
    """
    datetime_info = get_datetime_columns_info()
    if table_name in datetime_info:
        for col_info in datetime_info[table_name]:
            if col_info['column'].lower() == column_name.lower():
                return True, col_info['type']
    return False, None


def clear_schema_cache():
    """
    Clear the cached schema data.
    """
    if hasattr(st.cache_data, 'clear'):
        st.cache_data.clear()
    if 'datetime_columns_info' in st.session_state:
        del st.session_state['datetime_columns_info']


def get_table_info(table_name):
    """
    Get detailed information about a specific table.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        info_query = """
        SELECT 
            c.name AS column_name,
            t.name AS data_type,
            c.max_length,
            c.precision,
            c.scale,
            c.is_nullable,
            c.is_identity,
            dc.definition AS default_value
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        LEFT JOIN sys.default_constraints dc ON c.default_object_id = dc.object_id
        WHERE c.object_id = OBJECT_ID(?)
        ORDER BY c.column_id
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(info_query, (table_name,))
                return list(result)
        else:
            cursor = conn.cursor()
            cursor.execute(info_query, (table_name,))
            rows = cursor.fetchall()
            cursor.close()
            return rows

    except Exception as e:
        st.error(f"Error getting table info for {table_name}: {e}")
        return None


def verify_database_structure():
    """
    Verify that all required tables exist and have the correct structure.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "No database connection"

        required_tables = ['accounts_data', 'dormant_flags', 'dormant_ledger', 'insight_log', 'sql_query_history']
        missing_tables = []

        for table in required_tables:
            if hasattr(conn, 'connect'):
                with conn.connect() as connection:
                    exists = table_exists(connection, table)
            else:
                cursor = conn.cursor()
                exists = table_exists_direct(cursor, table)
                cursor.close()

            if not exists:
                missing_tables.append(table)

        if missing_tables:
            return False, f"Missing tables: {', '.join(missing_tables)}"
        else:
            return True, f"All {len(required_tables)} required tables exist"

    except Exception as e:
        return False, f"Error verifying database structure: {e}"


def get_table_schema(table_name):
    """
    Get detailed schema information for a specific table.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        schema_query = """
        SELECT 
            c.name AS column_name,
            t.name AS data_type,
            c.max_length,
            c.precision,
            c.scale,
            c.is_nullable,
            c.is_identity,
            CASE 
                WHEN pk.column_name IS NOT NULL THEN 'YES' 
                ELSE 'NO' 
            END AS is_primary_key,
            dc.definition AS default_value
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        LEFT JOIN (
            SELECT ku.table_name, ku.column_name
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
            INNER JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS ku
                ON tc.constraint_type = 'PRIMARY KEY' 
                AND tc.constraint_name = ku.constraint_name
        ) pk ON pk.table_name = ? AND pk.column_name = c.name
        LEFT JOIN sys.default_constraints dc ON c.default_object_id = dc.object_id
        WHERE c.object_id = OBJECT_ID(?)
        ORDER BY c.column_id
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(schema_query, (table_name, table_name))
                return list(result)
        else:
            cursor = conn.cursor()
            cursor.execute(schema_query, (table_name, table_name))
            rows = cursor.fetchall()
            cursor.close()
            return rows

    except Exception as e:
        st.error(f"Error getting schema for {table_name}: {e}")
        return None


def create_table_from_dataframe(df, table_name, primary_key=None):
    """
    Create a new table based on DataFrame structure.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "No database connection"

        # Generate CREATE TABLE statement
        create_sql = generate_create_table_sql(df, table_name, primary_key)

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                connection.execute(create_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(create_sql)
            conn.commit()
            cursor.close()

        return True, f"Table {table_name} created successfully"

    except Exception as e:
        return False, f"Error creating table {table_name}: {e}"


def generate_create_table_sql(df, table_name, primary_key=None):
    """
    Generate CREATE TABLE SQL statement from DataFrame.
    """
    columns_sql = []

    for col_name in df.columns:
        # Determine data type based on DataFrame column
        dtype = df[col_name].dtype

        if pd.api.types.is_integer_dtype(dtype):
            sql_type = "INT"
        elif pd.api.types.is_float_dtype(dtype):
            sql_type = "DECIMAL(18,2)"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            sql_type = "DATETIME2"
        elif pd.api.types.is_bool_dtype(dtype):
            sql_type = "NVARCHAR(50)"
        else:
            # Estimate string length
            max_len = df[col_name].astype(str).str.len().max()
            if max_len > 4000:
                sql_type = "NVARCHAR(MAX)"
            else:
                sql_type = f"NVARCHAR({max(max_len * 2, 50)})"

        # Add primary key constraint if specified
        if primary_key and col_name == primary_key:
            columns_sql.append(f"[{col_name}] {sql_type} NOT NULL PRIMARY KEY")
        else:
            columns_sql.append(f"[{col_name}] {sql_type}")

    create_sql = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns_sql) + "\n)"
    return create_sql


def drop_table(table_name):
    """
    Drop a table from the database.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "No database connection"

        drop_sql = f"DROP TABLE IF EXISTS {table_name}"

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                connection.execute(drop_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(drop_sql)
            conn.commit()
            cursor.close()

        return True, f"Table {table_name} dropped successfully"

    except Exception as e:
        return False, f"Error dropping table {table_name}: {e}"


def get_database_size_info():
    """
    Get information about database size and usage.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        size_query = """
        SELECT 
            DB_NAME() AS database_name,
            SUM(CAST(FILEPROPERTY(name, 'SpaceUsed') AS bigint) * 8192.) / 1024 / 1024 AS used_space_mb,
            SUM(CAST(size AS bigint) * 8192.) / 1024 / 1024 AS allocated_space_mb
        FROM sys.database_files
        WHERE type_desc = 'ROWS'
        """

        table_sizes_query = """
        SELECT 
            t.name AS table_name,
            p.rows AS row_count,
            (SUM(a.total_pages) * 8) / 1024.0 AS total_space_mb,
            (SUM(a.used_pages) * 8) / 1024.0 AS used_space_mb
        FROM sys.tables t
        INNER JOIN sys.indexes i ON t.object_id = i.object_id
        INNER JOIN sys.partitions p ON i.object_id = p.object_id AND i.index_id = p.index_id
        INNER JOIN sys.allocation_units a ON p.partition_id = a.container_id
        WHERE t.is_ms_shipped = 0
        GROUP BY t.name, p.rows
        ORDER BY used_space_mb DESC
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                db_size = pd.read_sql(size_query, connection)
                table_sizes = pd.read_sql(table_sizes_query, connection)
        else:
            db_size = pd.read_sql(size_query, conn)
            table_sizes = pd.read_sql(table_sizes_query, conn)

        return {
            'database_info': db_size,
            'table_sizes': table_sizes
        }

    except Exception as e:
        st.error(f"Error getting database size info: {e}")
        return None


def create_index(table_name, column_names, index_name=None):
    """
    Create an index on specified columns.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "No database connection"

        if index_name is None:
            index_name = f"IX_{table_name}_{'_'.join(column_names)}"

        columns_str = ', '.join([f"[{col}]" for col in column_names])
        create_index_sql = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                connection.execute(create_index_sql)
        else:
            cursor = conn.cursor()
            cursor.execute(create_index_sql)
            conn.commit()
            cursor.close()

        return True, f"Index {index_name} created successfully"

    except Exception as e:
        return False, f"Error creating index: {e}"


def get_table_indexes(table_name):
    """
    Get information about indexes on a table.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        indexes_query = """
        SELECT 
            i.name AS index_name,
            i.type_desc AS index_type,
            i.is_unique,
            i.is_primary_key,
            STRING_AGG(c.name, ', ') AS columns
        FROM sys.indexes i
        INNER JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id
        INNER JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id
        WHERE i.object_id = OBJECT_ID(?)
        GROUP BY i.name, i.type_desc, i.is_unique, i.is_primary_key
        ORDER BY i.index_id
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                result = connection.execute(indexes_query, (table_name,))
                return list(result)
        else:
            cursor = conn.cursor()
            cursor.execute(indexes_query, (table_name,))
            rows = cursor.fetchall()
            cursor.close()
            return rows

    except Exception as e:
        st.error(f"Error getting indexes for {table_name}: {e}")
        return None


def analyze_table_statistics(table_name):
    """
    Analyze table statistics and data distribution.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        # Get basic table stats
        stats_query = f"""
        SELECT 
            COUNT(*) AS total_rows,
            COUNT(DISTINCT *) AS unique_rows
        FROM {table_name}
        """

        # Get column statistics
        column_stats_query = f"""
        SELECT 
            c.name AS column_name,
            t.name AS data_type,
            c.is_nullable,
            CASE 
                WHEN t.name IN ('varchar', 'nvarchar', 'char', 'nchar') 
                THEN c.max_length 
                ELSE NULL 
            END AS max_length
        FROM sys.columns c
        INNER JOIN sys.types t ON c.user_type_id = t.user_type_id
        WHERE c.object_id = OBJECT_ID('{table_name}')
        ORDER BY c.column_id
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                basic_stats = pd.read_sql(stats_query, connection)
                column_stats = pd.read_sql(column_stats_query, connection)
        else:
            basic_stats = pd.read_sql(stats_query, conn)
            column_stats = pd.read_sql(column_stats_query, conn)

        return {
            'basic_stats': basic_stats,
            'column_stats': column_stats
        }

    except Exception as e:
        st.error(f"Error analyzing table {table_name}: {e}")
        return None


def export_table_schema(table_name, output_format='sql'):
    """
    Export table schema in various formats.
    """
    try:
        schema_info = get_table_schema(table_name)
        if not schema_info:
            return None, "Could not retrieve schema"

        if output_format.lower() == 'sql':
            # Generate CREATE TABLE statement
            columns = []
            for row in schema_info:
                col_name, data_type, max_length, precision, scale, is_nullable, is_identity, is_pk, default_val = row

                # Build column definition
                col_def = f"[{col_name}] {data_type}"

                if data_type in ('varchar', 'nvarchar', 'char', 'nchar') and max_length > 0:
                    length = max_length // 2 if data_type.startswith('n') else max_length
                    col_def += f"({length})"
                elif data_type in ('decimal', 'numeric') and precision:
                    col_def += f"({precision},{scale or 0})"

                if is_identity:
                    col_def += " IDENTITY(1,1)"

                if not is_nullable:
                    col_def += " NOT NULL"

                if is_pk == 'YES':
                    col_def += " PRIMARY KEY"

                if default_val:
                    col_def += f" DEFAULT {default_val}"

                columns.append(col_def)

            create_statement = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
            return create_statement, "SQL CREATE statement generated"

        elif output_format.lower() == 'json':
            import json
            schema_dict = {
                'table_name': table_name,
                'columns': []
            }

            for row in schema_info:
                col_name, data_type, max_length, precision, scale, is_nullable, is_identity, is_pk, default_val = row
                schema_dict['columns'].append({
                    'name': col_name,
                    'data_type': data_type,
                    'max_length': max_length,
                    'precision': precision,
                    'scale': scale,
                    'is_nullable': bool(is_nullable),
                    'is_identity': bool(is_identity),
                    'is_primary_key': is_pk == 'YES',
                    'default_value': default_val
                })

            return json.dumps(schema_dict, indent=2), "JSON schema exported"

        else:
            return None, f"Unsupported output format: {output_format}"

    except Exception as e:
        return None, f"Error exporting schema: {e}"


def backup_database_schema():
    """
    Create a backup of the entire database schema.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return None, "No database connection"

        # Get all user tables
        tables_query = """
        SELECT name FROM sys.tables WHERE is_ms_shipped = 0 ORDER BY name
        """

        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                tables_result = connection.execute(tables_query)
                table_names = [row[0] for row in tables_result]
        else:
            cursor = conn.cursor()
            cursor.execute(tables_query)
            table_names = [row[0] for row in cursor.fetchall()]
            cursor.close()

        # Export each table schema
        schema_backup = {
            'backup_timestamp': datetime.now().isoformat(),
            'database_name': 'Banking Compliance Database',
            'tables': {}
        }

        for table_name in table_names:
            sql_schema, _ = export_table_schema(table_name, 'sql')
            if sql_schema:
                schema_backup['tables'][table_name] = sql_schema

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"database_schema_backup_{timestamp}.json"

        import json
        with open(filename, 'w') as f:
            json.dump(schema_backup, f, indent=2)

        return filename, f"Schema backup saved to {filename}"

    except Exception as e:
        return None, f"Error backing up schema: {e}"


def restore_database_schema(backup_file):
    """
    Restore database schema from backup file.
    """
    try:
        import json

        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        conn = get_db_connection()
        if conn is None:
            return False, "No database connection"

        restored_tables = []
        errors = []

        for table_name, create_sql in backup_data['tables'].items():
            try:
                if hasattr(conn, 'connect'):
                    with conn.connect() as connection:
                        connection.execute(create_sql)
                else:
                    cursor = conn.cursor()
                    cursor.execute(create_sql)
                    conn.commit()
                    cursor.close()

                restored_tables.append(table_name)
            except Exception as e:
                errors.append(f"{table_name}: {str(e)}")

        if restored_tables and not errors:
            return True, f"Successfully restored {len(restored_tables)} tables"
        elif restored_tables and errors:
            return True, f"Restored {len(restored_tables)} tables with {len(errors)} errors: {'; '.join(errors)}"
        else:
            return False, f"Failed to restore any tables: {'; '.join(errors)}"

    except Exception as e:
        return False, f"Error restoring schema: {e}"
        return False


def table_exists_direct(cursor, table_name):
    """Check if table exists using direct cursor."""
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = 'dbo'
        """, (table_name,))
        result = cursor.fetchone()
        return result[0] > 0 if result else False
    except Exception as e:
        return False, f"Error verifying database structure: {e}"