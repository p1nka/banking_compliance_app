# database/connection.py
"""
Database connection handling for Azure SQL Database.
Streamlit Cloud-compatible version using pymssql.
"""

import os
import streamlit as st
import pandas as pd
import pymssql
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import time
from datetime import datetime

# Global connection pool and keep-alive settings
CONNECTION_POOL = {}
LAST_ACTIVITY = {}
CONNECTION_TIMEOUT = 10800  # 3 hours before forced reconnection
CONNECTION_KEEPALIVE = 300  # 5 minutes between keepalive pings
KEEPALIVE_ENABLED = True


@st.cache_resource(ttl="1h")
def get_db_connection():
    """
    Creates and returns a connection to the Azure SQL database.
    Optimized for Streamlit Cloud using pymssql with keep-alive support.
    Returns None if connection fails.
    """
    # Generate connection key for pooling
    conn_key = "main_db_connection"

    # Check if we have a valid cached connection
    if conn_key in CONNECTION_POOL and CONNECTION_POOL[conn_key] is not None:
        if is_connection_alive(conn_key):
            update_last_activity(conn_key)
            return CONNECTION_POOL[conn_key]
        else:
            # Connection is dead, remove it
            cleanup_connection(conn_key)

    # Create new connection
    connection = create_new_connection()

    if connection:
        # Store in pool and set activity time
        CONNECTION_POOL[conn_key] = connection
        update_last_activity(conn_key)

        # Start keep-alive if enabled
        if KEEPALIVE_ENABLED:
            schedule_keepalive(conn_key)

    return connection


def create_new_connection():
    """
    Create a new database connection with all fallback options.
    """
    try:
        # Get credentials from secrets first, then environment variables
        server, database, username, password, port = get_connection_credentials()

        if not all([server, database, username, password]):
            st.sidebar.error("❌ Database credentials missing")
            show_credentials_help()
            return None

        # Clean server name (remove protocol if present and .database.windows.net if missing)
        server = server.replace("https://", "").replace("http://", "")
        if not server.endswith(".database.windows.net"):
            if not server.endswith(".database.windows.net"):
                server = f"{server}.database.windows.net"

        st.sidebar.info(f"🔌 Connecting to: {server}")

        # Try pymssql connection with timeout
        try:
            connection = pymssql.connect(
                server=server,
                user=username,
                password=password,
                database=database,
                port=port,
                timeout=30,
                login_timeout=30,
                as_dict=False
            )

            # Test the connection with a simple query
            if test_connection_query(connection, "pymssql"):
                st.sidebar.success("✅ Connected via pymssql")
                return connection
            else:
                connection.close()
                raise Exception("Connection test query failed")

        except Exception as pymssql_error:
            st.sidebar.warning(f"⚠️ pymssql connection failed: {str(pymssql_error)[:100]}")

            # Try SQLAlchemy as fallback
            try:
                connection_string = f"mssql+pymssql://{username}:{quote_plus(password)}@{server}:{port}/{database}"

                engine = create_engine(
                    connection_string,
                    pool_pre_ping=True,
                    pool_recycle=300,
                    pool_size=5,
                    max_overflow=10,
                    connect_args={
                        "timeout": 30,
                        "login_timeout": 30,
                    }
                )

                # Test the SQLAlchemy connection
                if test_connection_query(engine, "SQLAlchemy"):
                    st.sidebar.success("✅ Connected via SQLAlchemy")
                    return engine
                else:
                    raise Exception("SQLAlchemy connection test failed")

            except Exception as sqlalchemy_error:
                st.sidebar.error(f"❌ All connection attempts failed")

                # Show detailed error information
                with st.sidebar.expander("🔧 Connection Errors"):
                    st.text(f"pymssql: {str(pymssql_error)[:200]}")
                    st.text(f"SQLAlchemy: {str(sqlalchemy_error)[:200]}")

                show_connection_troubleshooting()
                return None

    except Exception as e:
        st.sidebar.error(f"❌ Connection setup failed: {e}")
        show_connection_troubleshooting()
        return None


def get_connection_credentials():
    """
    Get database connection credentials from various sources.
    Returns (server, database, username, password, port)
    """
    server = None
    database = None
    username = None
    password = None
    port = 1433

    # Try Streamlit secrets first
    try:
        # Handle both possible key names in secrets
        server = st.secrets.get("DB_SERVER_NAME") or st.secrets.get("DB_SERVER")
        database = st.secrets.get("DB_NAME")
        username = st.secrets.get("DB_USERNAME")
        password = st.secrets.get("DB_PASSWORD")
        port = int(st.secrets.get("DB_PORT", "1433"))

        st.sidebar.info(f"📋 Using credentials from secrets.toml")
        st.sidebar.info(f"🗄️ Server: {server}")
        st.sidebar.info(f"🗄️ Database: {database}")
        st.sidebar.info(f"👤 Username: {username}")
        st.sidebar.info(f"🔌 Port: {port}")

    except Exception as e:
        st.sidebar.warning(f"⚠️ Failed to load from secrets: {e}")

    # Fallback to environment variables
    if not all([server, database, username, password]):
        server = server or os.getenv("DB_SERVER_NAME") or os.getenv("DB_SERVER")
        database = database or os.getenv("DB_NAME")
        username = username or os.getenv("DB_USERNAME")
        password = password or os.getenv("DB_PASSWORD")
        port = port or int(os.getenv("DB_PORT", "1433"))

    # Import config values as fallback
    if not server or not database:
        try:
            from config import DB_SERVER, DB_NAME, DB_PORT
            server = server or DB_SERVER
            database = database or DB_NAME
            port = port or DB_PORT
        except ImportError:
            pass

    return server, database, username, password, port


def test_connection_query(connection, connection_type):
    """
    Test connection with a simple query.
    Returns True if successful, False otherwise.
    """
    try:
        if connection_type == "SQLAlchemy":
            with connection.connect() as test_conn:
                result = test_conn.execute("SELECT 1")
                row = result.fetchone()
                return row and row[0] == 1
        else:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result and result[0] == 1
    except Exception as e:
        st.sidebar.error(f"Connection test failed: {e}")
        return False


def is_connection_alive(conn_key):
    """
    Check if a connection is still alive and responsive.
    """
    if conn_key not in CONNECTION_POOL or CONNECTION_POOL[conn_key] is None:
        return False

    connection = CONNECTION_POOL[conn_key]

    try:
        # Check connection age
        if conn_key in LAST_ACTIVITY:
            elapsed = (datetime.now() - LAST_ACTIVITY[conn_key]).total_seconds()
            if elapsed > CONNECTION_TIMEOUT:
                st.sidebar.info(f"🔄 Connection timeout ({elapsed:.0f}s), reconnecting...")
                return False

        # Test with a simple query
        if hasattr(connection, 'execute'):
            # SQLAlchemy engine
            with connection.connect() as test_conn:
                test_conn.execute("SELECT 1")
                return True
        else:
            # Direct pymssql connection
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True

    except Exception as e:
        st.sidebar.warning(f"⚠️ Connection became stale: {str(e)[:50]}")
        return False


def update_last_activity(conn_key):
    """
    Update the last activity timestamp for a connection.
    """
    LAST_ACTIVITY[conn_key] = datetime.now()


def cleanup_connection(conn_key):
    """
    Clean up a dead or stale connection.
    """
    if conn_key in CONNECTION_POOL:
        try:
            connection = CONNECTION_POOL[conn_key]
            if connection and hasattr(connection, 'close'):
                connection.close()
        except:
            pass
        del CONNECTION_POOL[conn_key]

    if conn_key in LAST_ACTIVITY:
        del LAST_ACTIVITY[conn_key]


def schedule_keepalive(conn_key):
    """
    Schedule keep-alive pings for the connection.
    This uses Streamlit's session state to track when to ping.
    """
    if 'keepalive_scheduler' not in st.session_state:
        st.session_state.keepalive_scheduler = {}

    st.session_state.keepalive_scheduler[conn_key] = datetime.now()


def perform_keepalive():
    """
    Perform keep-alive pings on all active connections.
    Call this periodically from your main app.
    """
    if not KEEPALIVE_ENABLED:
        return

    current_time = datetime.now()
    connections_pinged = 0

    for conn_key in list(CONNECTION_POOL.keys()):
        if conn_key in LAST_ACTIVITY:
            elapsed = (current_time - LAST_ACTIVITY[conn_key]).total_seconds()

            # Ping if it's been more than KEEPALIVE interval
            if elapsed > CONNECTION_KEEPALIVE:
                if ping_connection(conn_key):
                    connections_pinged += 1
                    update_last_activity(conn_key)
                else:
                    # Connection failed, mark for cleanup
                    cleanup_connection(conn_key)

    if connections_pinged > 0:
        st.sidebar.caption(f"🔄 Refreshed {connections_pinged} connection(s)")


def ping_connection(conn_key):
    """
    Send a keep-alive ping to a specific connection.
    Returns True if successful, False if connection should be cleaned up.
    """
    if conn_key not in CONNECTION_POOL or CONNECTION_POOL[conn_key] is None:
        return False

    connection = CONNECTION_POOL[conn_key]

    try:
        if hasattr(connection, 'execute'):
            # SQLAlchemy engine
            with connection.connect() as test_conn:
                test_conn.execute("SELECT 1")
                return True
        else:
            # Direct pymssql connection
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True

    except Exception as e:
        st.sidebar.warning(f"⚠️ Keep-alive failed for {conn_key}: {str(e)[:50]}")
        return False


def show_credentials_help():
    """Show help for setting up credentials."""
    st.sidebar.info("""
    **Required Configuration in .streamlit/secrets.toml:**
    ```toml
    DB_SERVER_NAME = "rahulsalpard.database.windows.net"
    DB_NAME = "rahulsalpard"
    DB_USERNAME = "agent123"
    DB_PASSWORD = "Aug@2025"
    DB_PORT = 1433
    ```

    **Note:** Remove http:// prefix from server name
    """)


def show_connection_troubleshooting():
    """
    Show troubleshooting information for connection issues.
    """
    with st.sidebar.expander("🔧 Troubleshooting", expanded=False):
        st.markdown("""
        **Common Issues:**

        1. **Server Name Format**
           - Should be: `servername.database.windows.net`
           - Remove `http://` or `https://` prefix

        2. **Firewall Rules**
           - Add your IP to Azure SQL firewall
           - Enable "Allow Azure services"

        3. **Credentials**
           - Verify username/password
           - Check authentication method

        4. **Network**
           - Port 1433 should be open
           - Check VPN/proxy settings

        **Expected Format:**
        Server: `rahulsalpard.database.windows.net`
        """)


def test_database_connection():
    """
    Test database connection and return status.
    Returns (success: bool, message: str)
    """
    try:
        conn = get_db_connection()
        if conn is None:
            return False, "No connection established"

        # Test with a simple query
        if hasattr(conn, 'execute'):
            # SQLAlchemy engine
            with conn.connect() as test_conn:
                result = test_conn.execute("SELECT GETDATE() as current_time, DB_NAME() as database_name")
                row = result.fetchone()
                return True, f"Connected to '{row[1]}' at {row[0]}"
        else:
            # Direct pymssql connection
            cursor = conn.cursor()
            cursor.execute("SELECT GETDATE(), DB_NAME()")
            row = cursor.fetchone()
            cursor.close()
            return True, f"Connected to '{row[1]}' at {row[0]}"

    except Exception as e:
        return False, f"Connection test failed: {str(e)[:200]}"


def execute_query(query, connection=None):
    """
    Execute SQL query with proper error handling.
    """
    if connection is None:
        connection = get_db_connection()

    if connection is None:
        st.error("❌ No database connection available")
        return None

    try:
        # Handle different connection types
        if hasattr(connection, 'execute'):
            # SQLAlchemy engine
            return pd.read_sql(query, connection)
        else:
            # Direct pymssql connection
            return pd.read_sql(query, connection)

    except Exception as e:
        st.error(f"❌ Query execution failed: {e}")
        return None


def force_reconnect():
    """
    Force a fresh reconnection by clearing all cached connections.
    """
    st.sidebar.info("🔄 Forcing database reconnection...")

    # Clear Streamlit cache
    get_db_connection.clear()

    # Clean up connection pool
    close_all_connections()

    # Get fresh connection
    conn = get_db_connection()

    if conn:
        st.sidebar.success("✅ Fresh connection established")
        return conn
    else:
        st.sidebar.error("❌ Failed to establish fresh connection")
        return None


def close_all_connections():
    """
    Close all database connections.
    """
    for conn_key, conn in CONNECTION_POOL.items():
        try:
            if conn and hasattr(conn, 'close'):
                conn.close()
        except:
            pass
    CONNECTION_POOL.clear()
    LAST_ACTIVITY.clear()


def ping_connections():
    """
    Ping all connections to keep them alive.
    Legacy function - use perform_keepalive() instead.
    """
    perform_keepalive()


def show_connection_status():
    """
    Display connection status in sidebar.
    """
    try:
        success, message = test_database_connection()
        if success:
            st.sidebar.success(f"✅ Database: {message}")
        else:
            st.sidebar.error(f"❌ Database: {message}")
    except Exception as e:
        st.sidebar.error(f"❌ Database connection test failed: {str(e)[:100]}")


def maintain_connection():
    """
    Main function to maintain database connection health.
    Add this to your main app's sidebar or call it periodically.
    """
    with st.sidebar:
        st.markdown("### 🔗 Database Connection")

        # Connection status
        success, message = test_database_connection()

        if success:
            st.success("✅ Connected")
            st.caption(f"Status: {message}")

            # Show last activity if available
            if 'main_db_connection' in LAST_ACTIVITY:
                last_ping = LAST_ACTIVITY['main_db_connection']
                elapsed = (datetime.now() - last_ping).total_seconds()

                if elapsed < 60:
                    st.caption(f"🟢 Active ({elapsed:.0f}s ago)")
                elif elapsed < 300:
                    st.caption(f"🟡 Idle ({elapsed / 60:.1f}m ago)")
                else:
                    st.caption(f"🔴 Stale ({elapsed / 60:.1f}m ago)")

        else:
            st.error("❌ Disconnected")
            st.caption(f"Error: {message}")

        # Control buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh", help="Refresh connection"):
                perform_keepalive()
                st.rerun()

        with col2:
            if st.button("🔁 Reconnect", help="Force new connection"):
                force_reconnect()
                st.rerun()

        # Keep-alive toggle
        global KEEPALIVE_ENABLED
        KEEPALIVE_ENABLED = st.checkbox(
            "Auto Keep-Alive",
            value=KEEPALIVE_ENABLED,
            help="Automatically ping database to maintain connection"
        )

        # Auto-perform keep-alive if enabled
        if KEEPALIVE_ENABLED and success:
            perform_keepalive()