import streamlit as st
import time
import pandas as pd
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
                    test_conn.commit()
            elif hasattr(conn, 'execute'):
                # SQLAlchemy connection
                conn.execute("SELECT 1 as test")
            else:
                # Direct connection (pymssql)
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
            # Try to insert sample data if table is empty
            insert_sample_data_if_empty()
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
            # Direct connection (pymssql)
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
            
            conn.commit()
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
        
        conn.commit()
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


def create_accounts_table_sqlalchemy_conn(conn):
    """Create accounts_data table using SQLAlchemy connection."""
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


def create_support_tables_sqlalchemy_conn(conn):
    """Create support tables using SQLAlchemy connection."""
    # Create dormant_flags table
    try:
        check_dormant = """
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME = 'dormant_flags' AND TABLE_SCHEMA = 'dbo'
        """
        result = conn.execute(check_dormant)
        if result.fetchone()[0] == 0:
            dormant_sql = """
            CREATE TABLE dormant_flags (
                id INT IDENTITY(1,1) PRIMARY KEY,
                account_id NVARCHAR(255) NOT NULL,
                flag_instruction NVARCHAR(MAX) NOT NULL,
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(dormant_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_flags table: {e}")
    
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
                natural_language_query NVARCHAR(MAX) NOT NULL,
                sql_query NVARCHAR(MAX) NOT NULL,
                timestamp DATETIME2 DEFAULT GETDATE()
            )
            """
            conn.execute(history_sql)
    except Exception as e:
        st.sidebar.warning(f"Could not create sql_query_history table: {e}")


def create_support_tables_direct(cursor):
    """Create support tables using direct cursor."""
    # Check and create dormant_flags table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'dormant_flags' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE dormant_flags (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    account_id NVARCHAR(255) NOT NULL,
                    flag_instruction NVARCHAR(MAX) NOT NULL,
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create dormant_flags table: {e}")
    
    # Check and create sql_query_history table
    try:
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES 
                          WHERE TABLE_NAME = 'sql_query_history' AND TABLE_SCHEMA = 'dbo')
            BEGIN
                CREATE TABLE sql_query_history (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    natural_language_query NVARCHAR(MAX) NOT NULL,
                    sql_query NVARCHAR(MAX) NOT NULL,
                    timestamp DATETIME2 DEFAULT GETDATE()
                )
            END
        """)
    except Exception as e:
        st.sidebar.warning(f"Could not create sql_query_history table: {e}")

def get_schema_sys_tables(conn):
    """Get schema using sys.tables and sys.columns (most reliable for Azure SQL)"""
    schema_info = {}
    
    try:
        # Try different connection types
        if hasattr(conn, 'connect'):
            # SQLAlchemy engine
            with conn.connect() as connection:
                return _execute_sys_tables_query(connection)
        elif hasattr(conn, 'execute'):
            # SQLAlchemy connection
            return _execute_sys_tables_query(conn)
        else:
            # Direct connection
            return _execute_sys_tables_query_direct(conn)
            
    except Exception as e:
        raise Exception(f"sys.tables query failed: {e}")


def _execute_sys_tables_query(conn):
    """Execute sys.tables query for SQLAlchemy connections"""
    schema_query = """
    SELECT 
        t.name AS table_name,
        c.name AS column_name,
        ty.name AS data_type,
        c.max_length,
        c.is_nullable
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
    schema_query = """
    SELECT 
        t.name AS table_name,
        c.name AS column_name,
        ty.name AS data_type,
        c.max_length,
        c.is_nullable
    FROM sys.tables t
    INNER JOIN sys.columns c ON t.object_id = c.object_id
    INNER JOIN sys.types ty ON c.user_type_id = ty.user_type_id
    WHERE t.type = 'U'
    ORDER BY t.name, c.column_id
    """
    
    cursor = conn.cursor()
    cursor.execute(schema_query)
    rows = cursor.fetchall()
    cursor.close()
    
    schema_info = {}
    for row in rows:
        table_name, column_name, data_type, max_length, is_nullable = row
        if table_name not in schema_info:
            schema_info[table_name] = []
        
        # Format data type with length if applicable
        if max_length and max_length > 0:
            formatted_type = f"{data_type}({max_length})"
        else:
            formatted_type = data_type
            
        nullable_info = "NULL" if is_nullable else "NOT NULL"
        schema_info[table_name].append((column_name, f"{formatted_type} {nullable_info}"))
    
    return schema_info


def get_schema_information_schema(conn):
    """Get schema using INFORMATION_SCHEMA views"""
    try:
        if hasattr(conn, 'connect'):
            with conn.connect() as connection:
                return _execute_information_schema_query(connection)
        elif hasattr(conn, 'execute'):
            return _execute_information_schema_query(conn)
        else:
            return _execute_information_schema_query_direct(conn)
    except Exception as e:
        raise Exception(f"INFORMATION_SCHEMA query failed: {e}")


def _execute_information_schema_query(conn):
    """Execute INFORMATION_SCHEMA query for SQLAlchemy connections"""
    # Get tables first
    tables_query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo'
    ORDER BY TABLE_NAME
    """
    
    tables_result = conn.execute(tables_query)
    tables = [row[0] for row in tables_result]
    
    schema_info = {}
    for table_name in tables:
        # Get columns for each table
        columns_query = """
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = ? AND TABLE_SCHEMA = 'dbo'
        ORDER BY ORDINAL_POSITION
        """
        
        columns_result = conn.execute(columns_query, (table_name,))
        columns = []
        
        for col_row in columns_result:
            col_name, data_type, is_nullable, max_length = col_row
            
            # Format data type
            if max_length:
                formatted_type = f"{data_type}({max_length})"
            else:
                formatted_type = data_type
                
            nullable_status = "NULL" if is_nullable == "YES" else "NOT NULL"
            columns.append((col_name, f"{formatted_type} {nullable_status}"))
        
        schema_info[table_name] = columns
    
    return schema_info


def _execute_information_schema_query_direct(conn):
    """Execute INFORMATION_SCHEMA query for direct connections"""
    cursor = conn.cursor()
    
    # Get tables first
    cursor.execute("""
        SELECT TABLE_NAME 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'dbo'
        ORDER BY TABLE_NAME
    """)
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_info = {}
    for table_name in tables:
        # Get columns for each table
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = 'dbo'
            ORDER BY ORDINAL_POSITION
        """, (table_name,))
        
        columns = []
        for col_row in cursor.fetchall():
            col_name, data_type, is_nullable, max_length = col_row
            
            if max_length:
                formatted_type = f"{data_type}({max_length})"
            else:
                formatted_type = data_type
                
            nullable_status = "NULL" if is_nullable == "YES" else "NOT NULL"
            columns.append((col_name, f"{formatted_type} {nullable_status}"))
        
        schema_info[table_name] = columns
    
    cursor.close()
    return schema_info


def get_schema_direct_query(conn):
    """Fallback method: try to discover tables by direct querying"""
    common_tables = ['accounts_data', 'dormant_flags', 'sql_query_history']
    schema_info = {}
    
    for table_name in common_tables:
        try:
            if hasattr(conn, 'connect'):
                with conn.connect() as connection:
                    test_query = f"SELECT TOP 0 * FROM {table_name}"
                    result = connection.execute(test_query)
                    columns = [(col, 'VARCHAR') for col in result.keys()]
                    schema_info[table_name] = columns
            elif hasattr(conn, 'execute'):
                test_query = f"SELECT TOP 0 * FROM {table_name}"
                result = conn.execute(test_query)
                columns = [(col, 'VARCHAR') for col in result.keys()]
                schema_info[table_name] = columns
            else:
                cursor = conn.cursor()
                cursor.execute(f"SELECT TOP 0 * FROM {table_name}")
                columns = [(desc[0], 'VARCHAR') for desc in cursor.description]
                schema_info[table_name] = columns
                cursor.close()
                
        except Exception:
            # Table doesn't exist or no access
            continue
    
    return schema_info if schema_info else None


def _format_schema_from_result(result):
    """Format schema query result into standard format"""
    schema_info = {}
    for row in result:
        table_name, column_name, data_type, max_length, is_nullable = row
        if table_name not in schema_info:
            schema_info[table_name] = []
        
        if max_length and max_length > 0:
            formatted_type = f"{data_type}({max_length})"
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
            ('Customer_ID', 'NVARCHAR(255) NOT NULL'),
            ('Account_Type', 'NVARCHAR(255) NOT NULL'),
            ('Currency', 'NVARCHAR(50) NULL'),
            ('Account_Creation_Date', 'DATE NULL'),
            ('Current_Balance', 'DECIMAL(18,2) NULL'),
            ('Date_Last_Cust_Initiated_Activity', 'DATE NULL'),
            ('Date_Last_Customer_Communication_Any_Type', 'DATE NULL'),
            ('Expected_Account_Dormant', 'NVARCHAR(10) NULL'),
            ('Customer_Address_Known', 'NVARCHAR(10) NULL'),
            ('Customer_Has_Active_Liability_Account', 'NVARCHAR(10) NULL')
        ],
        'dormant_flags': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('account_id', 'NVARCHAR(255) NOT NULL'),
            ('flag_instruction', 'NVARCHAR(MAX) NOT NULL'),
            ('timestamp', 'DATETIME2 NULL')
        ],
        'sql_query_history': [
            ('id', 'INT IDENTITY NOT NULL'),
            ('natural_language_query', 'NVARCHAR(MAX) NOT NULL'),
            ('sql_query', 'NVARCHAR(MAX) NOT NULL'),
            ('timestamp', 'DATETIME2 NULL')
        ]
    }


def test_database_operations():
    """Test basic database operations."""
    st.subheader("🧪 Database Operations Test")
    
    try:
        conn = get_db_connection()
        if conn is None:
            st.error("No database connection available")
            return
        
        # Test basic query
        try:
            if hasattr(conn, 'connect'):
                # SQLAlchemy engine
                with conn.connect() as test_conn:
                    result = test_conn.execute("SELECT @@VERSION as version, GETDATE() as current_time")
                    data = result.fetchone()
                    st.success("✅ Basic query successful (SQLAlchemy Engine)")
                    st.write(f"SQL Version: {data[0][:50]}...")
                    st.write(f"Current Time: {data[1]}")
            elif hasattr(conn, 'execute'):
                # SQLAlchemy connection
                result = conn.execute("SELECT @@VERSION as version, GETDATE() as current_time")
                data = result.fetchone()
                st.success("✅ Basic query successful (SQLAlchemy Connection)")
                st.write(f"SQL Version: {data[0][:50]}...")
                st.write(f"Current Time: {data[1]}")
            else:
                # Direct connection
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION as version, GETDATE() as current_time")
                data = cursor.fetchone()
                cursor.close()
                st.success("✅ Basic query successful (Direct Connection)")
                st.write(f"SQL Version: {data[0][:50]}...")
                st.write(f"Current Time: {data[1]}")
        except Exception as query_error:
            st.error(f"❌ Basic query failed: {query_error}")
            return
        
        # Test schema fetch with detailed error reporting
        st.write("**Testing Schema Retrieval Methods:**")
        
        schema_methods = [
            ("sys.tables", get_schema_sys_tables),
            ("INFORMATION_SCHEMA", get_schema_information_schema),
            ("Direct Query", get_schema_direct_query)
        ]
        
        working_method = None
        for method_name, method_func in schema_methods:
            try:
                schema = method_func(conn)
                if schema:
                    st.success(f"✅ {method_name} method works: {len(schema)} tables")
                    working_method = method_name
                    break
                else:
                    st.warning(f"⚠️ {method_name} method returned empty result")
            except Exception as method_error:
                st.error(f"❌ {method_name} method failed: {method_error}")
        
        # Test final schema retrieval
        st.write("**Final Schema Test:**")
        schema = get_db_schema()
        if schema:
            st.success(f"✅ Final schema retrieval successful: {len(schema)} tables")
            
            # Show table details
            for table_name, columns in schema.items():
                with st.expander(f"Table: {table_name} ({len(columns)} columns)"):
                    for col_name, col_type in columns:
                        st.write(f"- {col_name}: {col_type}")
        else:
            st.error("❌ Final schema retrieval failed")
            
    except Exception as e:
        st.error(f"❌ Database test failed: {e}")
        st.error(f"Full traceback: {traceback.format_exc()}")


def insert_sample_data_if_empty():
    """Insert sample data if accounts_data table is empty."""
    try:
        conn = get_db_connection()
        if conn is None:
            return False
        
        # Check if table has data
        count = 0
        try:
            if hasattr(conn, 'connect'):
                with conn.connect() as test_conn:
                    result = test_conn.execute("SELECT COUNT(*) FROM accounts_data")
                    count = result.fetchone()[0]
            elif hasattr(conn, 'execute'):
                result = conn.execute("SELECT COUNT(*) FROM accounts_data")
                count = result.fetchone()[0]
            else:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM accounts_data")
                count = cursor.fetchone()[0]
                cursor.close()
        except Exception as count_error:
            st.sidebar.warning(f"Could not check table data count: {count_error}")
            return False
        
        if count == 0:
            st.sidebar.info("Inserting sample data...")
            # Insert minimal sample data
            sample_data = [
                ('ACC001', 'CUST001', 'Savings', 'AED', '2020-01-01', 5000.00, '2021-01-01', '2021-01-01', 'Yes', 'No', 'No'),
                ('ACC002', 'CUST002', 'Current', 'AED', '2021-01-01', 15000.00, '2024-01-01', '2024-01-01', 'No', 'Yes', 'Yes'),
                ('ACC003', 'CUST003', 'Savings', 'USD', '2019-06-15', 25000.00, '2020-12-01', '2020-12-01', 'Yes', 'Yes', 'No'),
            ]
            
            insert_sql = """
            INSERT INTO accounts_data 
            (Account_ID, Customer_ID, Account_Type, Currency, Account_Creation_Date, 
             Current_Balance, Date_Last_Cust_Initiated_Activity, Date_Last_Customer_Communication_Any_Type,
             Expected_Account_Dormant, Customer_Address_Known, Customer_Has_Active_Liability_Account)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            try:
                if hasattr(conn, 'connect'):
                    with conn.connect() as insert_conn:
                        for row in sample_data:
                            insert_conn.execute(insert_sql, row)
                        insert_conn.commit()
                elif hasattr(conn, 'execute'):
                    for row in sample_data:
                        conn.execute(insert_sql, row)
                    conn.commit()
                else:
                    cursor = conn.cursor()
                    for row in sample_data:
                        cursor.execute(insert_sql, row)
                    conn.commit()
                    cursor.close()
                
                st.sidebar.success("✅ Sample data inserted")
                return True
            except Exception as insert_error:
                st.sidebar.warning(f"Sample data insertion failed: {insert_error}")
                return False
            
    except Exception as e:
        st.sidebar.warning(f"Sample data check/insertion failed: {e}")
        return False
    
    return False

def enhanced_get_schema_with_datetime_info(conn):
    """
    Enhanced schema detection that specifically handles datetime types properly.
    """
    try:
        # Enhanced query that specifically identifies datetime types
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
        
        # Track datetime columns specifically
        if type_category == 'DATETIME_TYPE':
            datetime_columns[table_name].append({
                'column': column_name,
                'type': data_type,
                'formatted_type': formatted_type,
                'precision': precision,
                'scale': scale
            })
    
    # Store datetime info in session state for later use
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
                'column': column_name,
                'type': data_type,
                'formatted_type': formatted_type,
                'precision': precision,
                'scale': scale
            })
    
    st.session_state['datetime_columns_info'] = datetime_columns
    return schema_info


def diagnose_datetime_issues():
    """
    Diagnostic function to identify datetime issues in the current setup.
    """
    st.subheader("🔍 DateTime Issues Diagnosis")
    
    try:
        conn = get_db_connection()
        if not conn:
            st.error("No database connection available for diagnosis")
            return
        
        # Test 1: Check actual datetime column types
        st.write("**Test 1: Checking Actual DateTime Column Types**")
        try:
            enhanced_schema = enhanced_get_schema_with_datetime_info(conn)
            if enhanced_schema:
                datetime_info = st.session_state.get('datetime_columns_info', {})
                if datetime_info:
                    for table, cols in datetime_info.items():
                        if cols:
                            st.write(f"Table `{table}` datetime columns:")
                            for col in cols:
                                st.write(f"  - {col['column']}: {col['formatted_type']}")
                else:
                    st.info("No datetime columns detected")
            else:
                st.warning("Could not retrieve enhanced schema information")
        except Exception as e:
            st.error(f"Enhanced schema check failed: {e}")
        
        # Test 2: Check sample datetime data
        st.write("**Test 2: Sample DateTime Data**")
        try:
            sample_query = "SELECT TOP 3 Account_Creation_Date, Date_Last_Cust_Initiated_Activity FROM accounts_data WHERE Account_Creation_Date IS NOT NULL"
            
            if hasattr(conn, 'connect'):
                with conn.connect() as test_conn:
                    result = test_conn.execute(sample_query)
                    rows = result.fetchall()
                    if rows:
                        st.write("Sample datetime values:")
                        for i, row in enumerate(rows):
                            st.write(f"Row {i+1}: {row[0]} | {row[1]}")
                    else:
                        st.info("No sample data available")
            else:
                cursor = conn.cursor()
                cursor.execute(sample_query)
                rows = cursor.fetchall()
                cursor.close()
                if rows:
                    st.write("Sample datetime values:")
                    for i, row in enumerate(rows):
                        st.write(f"Row {i+1}: {row[0]} | {row[1]}")
                else:
                    st.info("No sample data available")
                    
        except Exception as e:
            st.warning(f"Could not retrieve sample datetime data: {e}")
        
        # Test 3: Test pandas datetime parsing
        st.write("**Test 3: Pandas DateTime Parsing Test**")
        try:
            test_query = "SELECT TOP 5 Account_ID, Account_Creation_Date FROM accounts_data"
            
            # Test without parse_dates
            df_no_parse = pd.read_sql(test_query, conn)
            st.write("Without parse_dates:")
            st.write(df_no_parse.dtypes)
            
            # Test with parse_dates
            df_with_parse = pd.read_sql(test_query, conn, parse_dates=['Account_Creation_Date'])
            st.write("With parse_dates:")
            st.write(df_with_parse.dtypes)
            
        except Exception as e:
            st.warning(f"Pandas parsing test failed: {e}")
        
        # Test 4: Test common datetime queries
        st.write("**Test 4: Common DateTime Query Patterns**")
        datetime_test_queries = [
            ("Date comparison", "SELECT COUNT(*) FROM accounts_data WHERE Account_Creation_Date >= '2020-01-01'"),
            ("Date range", "SELECT COUNT(*) FROM accounts_data WHERE Account_Creation_Date BETWEEN '2020-01-01' AND '2024-01-01'"),
            ("Last 30 days", "SELECT COUNT(*) FROM accounts_data WHERE Account_Creation_Date >= DATEADD(DAY, -30, GETDATE())"),
        ]
        
        for test_name, query in datetime_test_queries:
            try:
                if hasattr(conn, 'connect'):
                    with conn.connect() as test_conn:
                        result = test_conn.execute(query)
                        count = result.fetchone()[0]
                        st.success(f"✅ {test_name}: {count} records")
                else:
                    cursor = conn.cursor()
                    cursor.execute(query)
                    count = cursor.fetchone()[0]
                    cursor.close()
                    st.success(f"✅ {test_name}: {count} records")
            except Exception as e:
                st.error(f"❌ {test_name} failed: {e}")
                
    except Exception as e:
        st.error(f"Diagnosis failed: {e}")

@st.cache_data(ttl=600)  # Cache for 10 minutes instead of 1 hour
def get_db_schema():
    """
    Get database schema with enhanced datetime detection and multiple fallback methods.
    """
    try:
        conn = get_db_connection()
        if conn is None:
            st.sidebar.warning("Cannot fetch schema: No database connection")
            return get_fallback_schema()

        # Try enhanced schema detection first
        try:
            enhanced_schema = enhanced_get_schema_with_datetime_info(conn)
            if enhanced_schema:
                st.sidebar.success(f"✅ Enhanced schema loaded: {len(enhanced_schema)} tables")
                return enhanced_schema
        except Exception as enhanced_error:
            st.sidebar.warning(f"Enhanced schema failed: {str(enhanced_error)[:100]}")

        # Fallback to original methods
        schema_methods = [
            ("sys.tables", get_schema_sys_tables),
            ("INFORMATION_SCHEMA", get_schema_information_schema),
            ("Direct Query", get_schema_direct_query)
        ]
        
        for method_name, method_func in schema_methods:
            try:
                schema_info = method_func(conn)
                if schema_info:
                    st.sidebar.success(f"✅ Schema loaded via {method_name}: {len(schema_info)} tables")
                    return schema_info
            except Exception as method_error:
                st.sidebar.warning(f"Schema method '{method_name}' failed: {str(method_error)[:100]}")
                continue
        
        # If all methods fail, return fallback schema
        st.sidebar.warning("⚠️ All schema methods failed, using fallback schema")
        return get_fallback_schema()

    except Exception as e:
        st.sidebar.error(f"Schema fetch error: {e}")
        return get_fallback_schema()
