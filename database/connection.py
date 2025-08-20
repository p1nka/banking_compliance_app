# database/connection.py
"""
Database connection handling for the application.
Streamlit Cloud-compatible version using pymssql.
"""

import os
import streamlit as st
import pandas as pd
import pymssql
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from datetime import datetime

# Global connection pool and keep-alive settings
CONNECTION_POOL = {}
LAST_ACTIVITY = {}
CONNECTION_TIMEOUT = 3600
CONNECTION_KEEPALIVE = 300
KEEPALIVE_ENABLED = True


def get_connection_credentials():
    """
    Get database connection credentials from various sources with a clear precedence:
    1. Streamlit Secrets (st.secrets) - for deployment
    2. Environment Variables - for local/docker
    3. config.py file - for local development fallback
    Returns a tuple: (server, database, username, password, port)
    """
    try:
        if "database" in st.secrets:
            creds = st.secrets["database"]
            if all(k in creds for k in ["server", "name", "username", "password"]):
                st.sidebar.info("🔑 Using Streamlit secrets")
                return (creds["server"], creds["name"], creds["username"], creds["password"], int(creds.get("port", 1433)))
    except (AttributeError, FileNotFoundError):
        pass

    env_creds = {"server": os.getenv("DB_SERVER"), "database": os.getenv("DB_NAME"), "username": os.getenv("DB_USERNAME"), "password": os.getenv("DB_PASSWORD"), "port": os.getenv("DB_PORT")}
    if all(v for k, v in env_creds.items() if k != "port"):
        st.sidebar.info("🔑 Using environment variables")
        return (env_creds["server"], env_creds["database"], env_creds["username"], env_creds["password"], int(env_creds["port"]) if env_creds["port"] else 1433)

    try:
        from config import DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD
        DB_PORT = getattr(__import__("config"), "DB_PORT", 1433)
        if all([DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD]):
            #st.sidebar.info("🔑 Using config.py")
            return (DB_SERVER, DB_NAME, DB_USERNAME, DB_PASSWORD, DB_PORT)
    except ImportError:
        pass

    return None, None, None, None, 1433


@st.cache_resource(ttl="1h")
def get_db_connection():
    """Creates and returns a connection to the Azure SQL database."""
    conn_key = "main_db_connection"
    if conn_key in CONNECTION_POOL and is_connection_alive(conn_key):
        update_last_activity(conn_key)
        return CONNECTION_POOL[conn_key]

    cleanup_connection(conn_key)
    connection = create_new_connection()
    if connection:
        CONNECTION_POOL[conn_key] = connection
        update_last_activity(conn_key)
    return connection


def create_new_connection():
    """Create a new database connection with fallback options."""
    try:
        server, database, username, password, port = get_connection_credentials()
        if not all([server, database, username, password]):
            st.sidebar.error("❌ Database credentials missing")
            show_credentials_help()
            return None

        server = server.replace("https://", "").replace("http://", "")
        #st.sidebar.info(f"🔌 Connecting to: {server}")

        try:
            connection = pymssql.connect(server=server, user=username, password=password, database=database, port=port, timeout=30, login_timeout=30, as_dict=False)
            if test_connection_query(connection, "pymssql"):
                #st.sidebar.success("✅ Connected via pymssql")
                return connection
            else:
                connection.close()
                raise Exception("pymssql connection test query failed")
        except Exception as pymssql_error:
            st.sidebar.warning(f"⚠️ pymssql failed: {str(pymssql_error)[:100]}")
            try:
                connection_string = f"mssql+pymssql://{username}:{quote_plus(password)}@{server}:{port}/{database}"
                engine = create_engine(connection_string, pool_pre_ping=True, pool_recycle=300, connect_args={"timeout": 30, "login_timeout": 30})
                if test_connection_query(engine, "SQLAlchemy"):
                    st.sidebar.success("✅ Connected via SQLAlchemy")
                    return engine
                else:
                    raise Exception("SQLAlchemy connection test failed")
            except Exception as sqlalchemy_error:
                st.sidebar.error("❌ All connection attempts failed")
                with st.sidebar.expander("🔧 Connection Errors"):
                    st.text(f"pymssql: {str(pymssql_error)[:200]}")
                    st.text(f"SQLAlchemy: {str(sqlalchemy_error)[:200]}")
                show_connection_troubleshooting()
                return None
    except Exception as e:
        st.sidebar.error(f"❌ Connection setup failed: {e}")
        show_connection_troubleshooting()
        return None


def test_connection_query(connection, connection_type):
    """Test connection with a simple query."""
    try:
        if connection_type == "SQLAlchemy":
            with connection.connect() as conn:
                result = conn.execute("SELECT 1").fetchone()
                return result and result[0] == 1
        else:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                return result and result[0] == 1
    except:
        return False


def is_connection_alive(conn_key):
    """Check if a connection is still alive."""
    if conn_key not in CONNECTION_POOL or not CONNECTION_POOL[conn_key]:
        return False
    connection = CONNECTION_POOL[conn_key]
    try:
        if conn_key in LAST_ACTIVITY and (datetime.now() - LAST_ACTIVITY[conn_key]).total_seconds() > CONNECTION_TIMEOUT:
            st.sidebar.info("🔄 Connection timeout, reconnecting...")
            return False
        conn_type = "SQLAlchemy" if hasattr(connection, 'execute') else "pymssql"
        return test_connection_query(connection, conn_type)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Connection became stale: {str(e)[:50]}")
        return False


def update_last_activity(conn_key):
    LAST_ACTIVITY[conn_key] = datetime.now()


def cleanup_connection(conn_key):
    if conn_key in CONNECTION_POOL:
        try:
            CONNECTION_POOL.pop(conn_key, None).close()
        except: pass
    LAST_ACTIVITY.pop(conn_key, None)


def perform_keepalive():
    """Perform keep-alive pings on active connections."""
    if not KEEPALIVE_ENABLED: return
    for conn_key in list(CONNECTION_POOL.keys()):
        if conn_key in LAST_ACTIVITY and (datetime.now() - LAST_ACTIVITY[conn_key]).total_seconds() > CONNECTION_KEEPALIVE:
            if is_connection_alive(conn_key):
                update_last_activity(conn_key)
            else:
                cleanup_connection(conn_key)


def execute_query(query: str, connection=None):
    """Execute a SQL query and return a pandas DataFrame."""
    connection = connection or get_db_connection()
    if not connection:
        st.error("❌ No database connection available.")
        return None
    try:
        return pd.read_sql(query, connection)
    except Exception as e:
        st.error(f"❌ Query execution failed: {e}")
        return None


def close_all_connections():
    """Safely close all connections in the pool."""
    for conn_key in list(CONNECTION_POOL.keys()):
        cleanup_connection(conn_key)

# ADDED THIS FUNCTION BACK FOR COMPATIBILITY
def ping_connections():
    """
    Ping all connections to keep them alive.
    Legacy function - use perform_keepalive() instead.
    """
    perform_keepalive()


def force_reconnect():
    """Force a fresh reconnection."""
    st.sidebar.info("🔄 Forcing database reconnection...")
    get_db_connection.clear()
    close_all_connections()
    conn = get_db_connection()
    if conn:
        st.sidebar.success("✅ Fresh connection established")
    else:
        st.sidebar.error("❌ Failed to establish fresh connection")
    return conn

# --- UI HELPER FUNCTIONS ---

def wakeup_connection():
    """Wake up and refresh the database connection."""
    st.sidebar.info("🔄 Waking up database connection...")
    perform_keepalive()
    conn = get_db_connection()
    if conn:
        st.sidebar.success("✅ Database connection active")
    else:
        st.sidebar.error("❌ Failed to wake up database connection")
    return conn

def test_database_connection():
    """Test database connection and return status."""
    try:
        conn = get_db_connection()
        if not conn: return False, "No connection established"
        if hasattr(conn, 'execute'):
            with conn.connect() as test_conn:
                row = test_conn.execute("SELECT GETDATE(), DB_NAME()").fetchone()
        else:
            with conn.cursor() as cursor:
                cursor.execute("SELECT GETDATE(), DB_NAME()")
                row = cursor.fetchone()
        return True, f"Connected to {row[1]} at {row[0]}"
    except Exception as e:
        return False, f"Connection test failed: {str(e)[:200]}"

def show_connection_status():
    """Display connection status in sidebar."""
    success, message = test_database_connection()
    if success:
        st.sidebar.success(f"✅ Database: {message}")
    else:
        st.sidebar.error(f"❌ Database connection failed: {message}")

def maintain_connection():
    """Main function to display and maintain database connection health in the UI."""
    with st.sidebar:
        st.markdown("### 🔗 Database Connection")
        success, message = test_database_connection()
        if success:
            st.success("✅ Connected")
            st.caption(f"Status: {message}")
        else:
            st.error("❌ Disconnected")
            st.caption(f"Error: {message}")

        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh", help="Refresh connection"):
            wakeup_connection()
            st.rerun()
        if col2.button("🔁 Reconnect", help="Force new connection"):
            force_reconnect()
            st.rerun()

        global KEEPALIVE_ENABLED
        KEEPALIVE_ENABLED = st.checkbox("Auto Keep-Alive", value=KEEPALIVE_ENABLED, help="Automatically ping database to maintain connection")
        if KEEPALIVE_ENABLED and success:
            perform_keepalive()

def show_credentials_help():
    """Show help for setting up credentials."""
    with st.sidebar.expander("🔑 Setup Credentials"):
        st.markdown("""
        Configure credentials via:
        1. **Streamlit Secrets** (Recommended)
        2. **Environment Variables**
        3. `config.py` file
        """)

def show_connection_troubleshooting():
    """Show troubleshooting info."""
    with st.sidebar.expander("🔧 Troubleshooting"):
        st.markdown("""
        - **Firewall Rules**: Check your IP in Azure SQL firewall.
        - **Credentials**: Verify server, db, user, and password.
        - **Network**: Ensure port 1433 is open.
        """)