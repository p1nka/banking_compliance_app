import streamlit as st
import time
import pandas as pd
from datetime import datetime

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
            # Test the connection first
            if hasattr(conn, 'execute'):
                # SQLAlchemy engine
                with conn.connect() as test_conn:
                    test_conn.execute("SELECT 1")
            else:
                # Direct connection
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
        except Exception as test_error:
            st.sidebar.warning(f"⚠️ Database connection test failed: {test_error}")
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
        if hasattr(conn, 'execute'):
            # SQLAlchemy engine - use connection context
            return create_tables_sqlalchemy(conn)
        else:
            # Direct connection (pymssql)
            return create_tables_direct(conn)
    except Exception as e:
        st.sidebar.error(f"Table creation error: {e}")
        return False


def create_tables_sqlalchemy(engine):
    """Create tables using SQLAlchemy engine."""
    try:
        with engine.connect() as conn:
            # Check if main table exists
            check_query = """
            SELECT COUNT(*) as table_count 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'accounts_data'
            """
            
            result = conn.execute(check_query)
            table_exists = result.fetchone()[0] > 0
            
            if not table_exists:
                st.sidebar.info("Creating accounts_data table...")
                create_accounts_table_sqlalchemy(conn)
            
            # Create other essential tables
            create_support_tables_sqlalchemy(conn)
            
            conn.commit()
            return True
            
    except Exception as e:
        st.sidebar.error(f"SQLAlchemy table creation failed: {e}")
        return False


def create_tables_direct(conn):
    """Create tables using direct connection."""
    try:
        cursor = conn.cursor()
        
        # Check if main table exists
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = 'accounts_data'
        """)
        
        table_exists = cursor.fetchone()[0] > 0
        
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


def create_accounts_table_sqlalchemy(conn):
    """Create accounts_data table using SQLAlchemy."""
    create_sql = """
    CREATE TABLE accounts_data (
        Account_ID NVARCHAR(255) NOT NULL PRIMARY KEY,
        Customer_ID NVARCHAR(255) NOT NULL,
        Account_Type NVARCHAR(255) NOT NULL,
        Currency NVARCHAR(50),
        Account_Creation_Date DATE,
        Current_Balance DECIMAL(18, 2),
        Date_Last_Cust_Initiated_Activity DATE,
        Date_Last_Customer_Communication_Any_Type DATE,
        Expected_Account_Dormant NVARCHAR(10) DEFAULT 'No',
        Customer_Address_Known NVARCHAR(10) DEFAULT 'Unknown',
        Customer_Has_Active_Liability_Account NVARCHAR(10) DEFAULT 'Unknown'
    )
    """
    conn.execute(create_sql)


def create_accounts_table_direct(cursor):
    """Create accounts_data table using direct cursor."""
    create_sql = """
    CREATE TABLE accounts_data (
        Account_ID NVARCHAR(255) NOT NULL PRIMARY KEY,
        Customer_ID NVARCHAR(255) NOT NULL,
        Account_Type NVARCHAR(255) NOT NULL,
        Currency NVARCHAR(50),
        Account_Creation_Date DATE,
        Current_Balance DECIMAL(18, 2),
        Date_Last_Cust_Initiated_Activity DATE,
        Date_Last_Customer_Communication_Any_Type DATE,
        Expected_Account_Dormant NVARCHAR(10) DEFAULT 'No',
        Customer_Address_Known NVARCHAR(10) DEFAULT 'Unknown',
        Customer_Has_Active_Liability_Account NVARCHAR(10) DEFAULT 'Unknown'
    )
    """
    cursor.execute(create_sql)


def create_support_tables_sqlalchemy(conn):
    """Create support tables using SQLAlchemy."""
    tables_sql = [
        """
        CREATE TABLE IF NOT EXISTS dormant_flags (
            id INT IDENTITY(1,1) PRIMARY KEY,
            account_id NVARCHAR(255) NOT NULL,
            flag_instruction NVARCHAR(MAX) NOT NULL,
            timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sql_query_history (
            id INT IDENTITY(1,1) PRIMARY KEY,
            natural_language_query NVARCHAR(MAX) NOT NULL,
            sql_query NVARCHAR(MAX) NOT NULL,
            timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    for sql in tables_sql:
        try:
            conn.execute(sql.replace("CREATE TABLE IF NOT EXISTS", 
                        "IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = '{}') CREATE TABLE".format(
                            sql.split()[5])))  # Extract table name
        except:
            # Ignore if table already exists
            pass


def create_support_tables_direct(cursor):
    """Create support tables using direct cursor."""
    # Check and create dormant_flags table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags')
            BEGIN
                CREATE TABLE dormant_flags (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    account_id NVARCHAR(255) NOT NULL,
                    flag_instruction NVARCHAR(MAX) NOT NULL,
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
                )
            END
        """)
    except:
        pass
    
    # Check and create sql_query_history table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'sql_query_history')
            BEGIN
                CREATE TABLE sql_query_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    natural_language_query NVARCHAR(MAX) NOT NULL,
                    sql_query NVARCHAR(MAX) NOT NULL,
                    timestamp DATETIME2 DEFAULT CURRENT_TIMESTAMP
                )
            END
        """)
    except:
        pass


@st.cache_data(ttl="1h")
def get_db_schema():
    """
    Get database schema with enhanced error handling.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            st.sidebar.warning("Cannot fetch schema: No database connection")
            return None

        schema_info = {}
        
        if hasattr(conn, 'execute'):
            # SQLAlchemy engine
            schema_info = get_schema_sqlalchemy(conn)
        else:
            # Direct connection
            schema_info = get_schema_direct(conn)
        
        if schema_info:
            st.sidebar.info(f"✅ Schema loaded: {len(schema_info)} tables")
        else:
            st.sidebar.warning("⚠️ No tables found in schema")
            
        return schema_info

    except Exception as e:
        st.sidebar.error(f"Schema fetch error: {e}")
        return None


def get_schema_sqlalchemy(engine):
    """Get schema using SQLAlchemy engine."""
    schema_info = {}
    
    try:
        with engine.connect() as conn:
            # Get tables
            tables_query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
            tables_result = conn.execute(tables_query)
            tables = [row[0] for row in tables_result]
            
            for table_name in tables:
                # Get columns for each table
                columns_query = """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
                """
                columns_result = conn.execute(columns_query, (table_name,))
                
                columns = []
                for col_row in columns_result:
                    col_name, data_type, is_nullable = col_row
                    nullable_status = "NULL" if is_nullable == "YES" else "NOT NULL"
                    columns.append((col_name, f"{data_type} {nullable_status}"))
                
                schema_info[table_name] = columns
                
    except Exception as e:
        st.sidebar.error(f"SQLAlchemy schema error: {e}")
        
    return schema_info


def get_schema_direct(conn):
    """Get schema using direct connection."""
    schema_info = {}
    
    try:
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table_name in tables:
            # Get columns for each table
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """, (table_name,))
            
            columns = []
            for col_row in cursor.fetchall():
                col_name, data_type, is_nullable = col_row
                nullable_status = "NULL" if is_nullable == "YES" else "NOT NULL"
                columns.append((col_name, f"{data_type} {nullable_status}"))
            
            schema_info[table_name] = columns
        
        cursor.close()
        
    except Exception as e:
        st.sidebar.error(f"Direct schema error: {e}")
        
    return schema_info


def test_database_operations():
    """Test basic database operations."""
    st.subheader("🧪 Database Operations Test")
    
    try:
        conn = get_db_connection()
        if conn is None:
            st.error("No database connection available")
            return
        
        # Test basic query
        if hasattr(conn, 'execute'):
            # SQLAlchemy
            with conn.connect() as test_conn:
                result = test_conn.execute("SELECT @@VERSION as version, GETDATE() as current_time")
                data = result.fetchone()
                st.success("✅ Basic query successful")
                st.write(f"SQL Version: {data[0][:50]}...")
                st.write(f"Current Time: {data[1]}")
        else:
            # Direct connection
            cursor = conn.cursor()
            cursor.execute("SELECT @@VERSION as version, GETDATE() as current_time")
            data = cursor.fetchone()
            cursor.close()
            st.success("✅ Basic query successful")
            st.write(f"SQL Version: {data[0][:50]}...")
            st.write(f"Current Time: {data[1]}")
        
        # Test schema fetch
        schema = get_db_schema()
        if schema:
            st.success(f"✅ Schema fetch successful: {len(schema)} tables")
            
            # Show table list
            table_list = list(schema.keys())
            st.write("**Available Tables:**")
            for table in table_list:
                st.write(f"- {table} ({len(schema[table])} columns)")
        else:
            st.warning("⚠️ Schema fetch returned empty result")
            
    except Exception as e:
        st.error(f"❌ Database test failed: {e}")


def insert_sample_data_if_empty():
    """Insert sample data if accounts_data table is empty."""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        
        # Check if table has data
        if hasattr(conn, 'execute'):
            with conn.connect() as test_conn:
                result = test_conn.execute("SELECT COUNT(*) FROM accounts_data")
                count = result.fetchone()[0]
        else:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM accounts_data")
            count = cursor.fetchone()[0]
            cursor.close()
        
        if count == 0:
            st.sidebar.info("Inserting sample data...")
            # Insert minimal sample data
            sample_data = [
                ('ACC001', 'CUST001', 'Savings', 'AED', '2020-01-01', 5000.00, '2021-01-01', '2021-01-01', 'Yes', 'No', 'No'),
                ('ACC002', 'CUST002', 'Current', 'AED', '2021-01-01', 15000.00, '2024-01-01', '2024-01-01', 'No', 'Yes', 'Yes'),
            ]
            
            insert_sql = """
            INSERT INTO accounts_data 
            (Account_ID, Customer_ID, Account_Type, Currency, Account_Creation_Date, 
             Current_Balance, Date_Last_Cust_Initiated_Activity, Date_Last_Customer_Communication_Any_Type,
             Expected_Account_Dormant, Customer_Address_Known, Customer_Has_Active_Liability_Account)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            if hasattr(conn, 'execute'):
                with conn.connect() as insert_conn:
                    for row in sample_data:
                        insert_conn.execute(insert_sql, row)
                    insert_conn.commit()
            else:
                cursor = conn.cursor()
                for row in sample_data:
                    cursor.execute(insert_sql, row)
                conn.commit()
                cursor.close()
            
            st.sidebar.success("✅ Sample data inserted")
            return True
            
    except Exception as e:
        st.sidebar.warning(f"Sample data insertion failed: {e}")
        return False
    
    return False
