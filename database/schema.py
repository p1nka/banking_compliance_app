# --- START OF CORRECTED FILE schema.py ---

import streamlit as st
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import traceback

from database.connection import get_db_connection
from config import DB_NAME, DB_SERVER


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
            # Test the connection first with proper error handling
            if hasattr(conn, 'connect'):
                # SQLAlchemy engine
                with conn.connect() as test_conn:
                    result = test_conn.execute("SELECT 1 as test")
            elif hasattr(conn, 'execute'):
                # SQLAlchemy connection
                conn.execute("SELECT 1 as test")
            else:
                # Direct connection (pyodbc)
                cursor = conn.cursor()
                cursor.execute("SELECT 1 as test")
                cursor.fetchone()
                cursor.close()

            st.sidebar.success("✅ Database connection verified")
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
            # Direct connection (pyodbc)
            return create_tables_direct(conn)
    except Exception as e:
        st.sidebar.error(f"Table creation error: {e}")
        return False


def create_tables_sqlalchemy_engine(engine):
    """Create tables using SQLAlchemy engine."""
    try:
        with engine.connect() as conn:
            # Check if main table exists using proper Azure SQL syntax
            check_query = """
            SELECT COUNT(*) as table_count 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'accounts_data' AND TABLE_SCHEMA = 'dbo'
            """

            result = conn.execute(check_query)
            table_exists = result.fetchone()[0] > 0

            if not table_exists:
                st.sidebar.info("Creating accounts_data table...")
                create_accounts_table_sqlalchemy_conn(conn)

            # Create other essential tables
            create_support_tables_sqlalchemy_conn(conn)

            # Use .commit() if available on the raw connection for engines that need it
            if hasattr(conn.connection, 'commit'):
                conn.connection.commit()
            return True

    except Exception as e:
        st.sidebar.error(f"SQLAlchemy engine table creation failed: {e}")
        st.sidebar.error(f"Full error: {traceback.format_exc()}")
        return False


def create_tables_sqlalchemy_connection(conn):
    """Create tables using SQLAlchemy connection."""
    try:
        # Check if main table exists
        check_query = """
        SELECT COUNT(*) as table_count 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'accounts_data' AND TABLE_SCHEMA = 'dbo'
        """

        result = conn.execute(check_query)
        table_exists = result.fetchone()[0] > 0

        if not table_exists:
            st.sidebar.info("Creating accounts_data table...")
            create_accounts_table_sqlalchemy_conn(conn)

        # Create other essential tables
        create_support_tables_sqlalchemy_conn(conn)

        if hasattr(conn.connection, 'commit'):
            conn.connection.commit()
        return True

    except Exception as e:
        st.sidebar.error(f"SQLAlchemy connection table creation failed: {e}")
        return False


def create_tables_direct(conn):
    """Create tables using direct connection."""
    try:
        cursor = conn.cursor()

        # Check if main table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'accounts_data' AND TABLE_SCHEMA = 'dbo'
        """)

        result = cursor.fetchone()
        table_exists = result[0] > 0 if result else False

        if not table_exists:
            st.sidebar.info("Creating accounts_data table...")
            create_accounts_table_direct(cursor)

        # Create other essential tables
        create_support_tables_direct(cursor)

        conn.commit()
        cursor.close()
        return True

    except Exception as e:
        st.sidebar.error(f"Direct table creation failed: {e}")
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
    # FIX: Changed table name from dormant_flags to dormant_ledger
    try:
        check_dormant = """
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'dormant_ledger' AND TABLE_SCHEMA = 'dbo'
        """
        result = conn.execute(check_dormant)
        if result.fetchone()[0] == 0:
            dormant_sql = """
            CREATE TABLE dormant_ledger (
                id INT IDENTITY(1,1) PRIMARY KEY,
                account_id NVARCHAR(255) NOT NULL,
                flag_instruction NVARCHAR(MAX) NOT NULL,
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(dormant_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_ledger table: {e}")

    # Create sql_query_history table
    try:
        check_history = """
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'sql_query_history' AND TABLE_SCHEMA = 'dbo'
        """
        result = conn.execute(check_history)
        if result.fetchone()[0] == 0:
            history_sql = """
            CREATE TABLE sql_query_history (
                id INT IDENTITY(1,1) PRIMARY KEY,
                query_by NVARCHAR(255),
                query_text NVARCHAR(MAX),
                execution_time FLOAT,
                created_at DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(history_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create sql_query_history table: {e}")


def create_support_tables_direct(cursor):
    """Create support tables using direct cursor."""
    # FIX: Changed table name from dormant_flags to dormant_ledger
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'dormant_ledger' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE dormant_ledger (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    account_id NVARCHAR(255) NOT NULL,
                    flag_instruction NVARCHAR(MAX) NOT NULL,
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_ledger table: {e}")

    # Check and create sql_query_history table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'sql_query_history' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE sql_query_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    query_by NVARCHAR(255),
                    query_text NVARCHAR(MAX),
                    execution_time FLOAT,
                    created_at DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create sql_query_history table: {e}")


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
    # This query is kept simpler for fallback, the enhanced one is primary
    schema_query = "SELECT t.name AS table_name, c.name AS column_name, ty.name AS data_type, c.max_length, c.is_nullable FROM sys.tables t INNER JOIN sys.columns c ON t.object_id = c.object_id INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id WHERE t.type = 'U' ORDER BY t.name, c.column_id"
    result = conn.execute(schema_query)
    return _format_schema_from_result(result)


def _execute_sys_tables_query_direct(conn):
    """Execute sys.tables query for direct connections"""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT t.name AS table_name, c.name AS column_name, ty.name AS data_type, c.max_length, c.is_nullable FROM sys.tables t INNER JOIN sys.columns c ON t.object_id = c.object_id INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id WHERE t.type = 'U' ORDER BY t.name, c.column_id")
    rows = cursor.fetchall()
    cursor.close()
    return _format_schema_from_result(rows)


def get_schema_direct_query(conn):
    """Fallback method: try to discover tables by direct querying"""
    # FIX: Changed table name from dormant_flags to dormant_ledger
    common_tables = ['accounts_data', 'dormant_ledger', 'sql_query_history']
    schema_info = {}
    for table_name in common_tables:
        try:
            if hasattr(conn, 'connect'):
                with conn.connect() as connection:
                    result = connection.execute(f"SELECT TOP 0 * FROM {table_name}")
                    schema_info[table_name] = [(col, 'VARCHAR') for col in result.keys()]
            else:
                cursor = conn.cursor()
                cursor.execute(f"SELECT TOP 0 * FROM {table_name}")
                schema_info[table_name] = [(desc[0], 'VARCHAR') for desc in cursor.description]
                cursor.close()
        except Exception:
            continue
    return schema_info if schema_info else None


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
    # FIX: Changed table name from dormant_flags to dormant_ledger
    return {
        'accounts_data': [
            ('Account_ID', 'NVARCHAR(255) NOT NULL'),
            ('Customer_ID', 'NVARCHAR(255)'), ('Account_Type', 'NVARCHAR(255)'),
            ('Currency', 'NVARCHAR(50)'), ('Account_Creation_Date', 'DATE'),
            ('Current_Balance', 'DECIMAL(18, 2)'), ('Date_Last_Bank_Initiated_Activity', 'DATE'),
            ('Date_Last_Customer_Communication_Any_Type', 'DATE'), ('FTD_Maturity_Date', 'DATE'),
            ('FTD_Auto_Renewal', 'NVARCHAR(50)'), ('Date_Last_FTD_Renewal_Claim_Request', 'DATE'),
            ('Inv_Maturity_Redemption_Date', 'DATE'), ('SDB_Charges_Outstanding', 'NVARCHAR(50)'),
            ('Date_SDB_Charges_Became_Outstanding', 'DATE'), ('SDB_Tenant_Communication_Received', 'NVARCHAR(50)'),
            ('Unclaimed_Item_Trigger_Date', 'DATE'), ('Unclaimed_Item_Amount', 'DECIMAL(18, 2)'),
            ('Date_Last_Cust_Initiated_Activity', 'DATE'),
            ('Bank_Contact_Attempted_Post_Dormancy_Trigger', 'NVARCHAR(50)'),
            ('Date_Last_Bank_Contact_Attempt', 'DATE'), ('Customer_Responded_to_Bank_Contact', 'NVARCHAR(50)'),
            ('Date_Claim_Received', 'DATE'), ('Claim_Successful', 'NVARCHAR(50)'),
            ('Amount_Paid_on_Claim', 'DECIMAL(18, 2)'), ('Scenario_Notes', 'NVARCHAR(MAX)'),
            ('Customer_Address_Known', 'NVARCHAR(50)'), ('Customer_Has_Active_Liability_Account', 'NVARCHAR(50)'),
            ('Customer_Has_Litigation_Regulatory_Reqs', 'NVARCHAR(50)'),
            ('Holder_Has_Activity_On_Any_Other_Account', 'NVARCHAR(50)'),
            ('Is_Asset_Only_Customer_Type', 'NVARCHAR(50)'), ('Expected_Account_Dormant', 'NVARCHAR(50)'),
            ('Expected_Requires_Article_3_Process', 'NVARCHAR(50)'), ('Expected_Transfer_to_CB_Due', 'NVARCHAR(50)')
        ],
        'dormant_ledger': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('account_id', 'NVARCHAR(255) NOT NULL'),
            ('flag_instruction', 'NVARCHAR(MAX) NOT NULL'),
            ('timestamp', 'DATETIME2 NULL')
        ],
        'sql_query_history': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('query_by', 'NVARCHAR(255)'),
            ('query_text', 'NVARCHAR(MAX)'),
            ('execution_time', 'FLOAT'),
            ('created_at', 'DATETIME2')
        ]
    }


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

# --- END OF CORRECTED FILE schema.py ---