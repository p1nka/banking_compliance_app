# database/connection.py
"""
Database connection handling for the application.
"""

import os
import pyodbc
import streamlit as st
import time
import re
from datetime import datetime
from config import DB_SERVER, DB_NAME, DB_PORT

# Global connection pool to reuse connections
CONNECTION_POOL = {}
LAST_ACTIVITY = {}
CONNECTION_TIMEOUT = 3600  # 1 hour timeout for connections
CONNECTION_KEEPALIVE = 300  # 5 minutes keepalive ping


@st.cache_resource(ttl="1h")
def get_db_connection():
    """
    Creates and returns a connection to the Azure SQL database using the credentials
    in st.secrets or environment variables, and global DB_SERVER/DB_NAME.
    Returns None if connection fails or credentials are not found.

    This function is cached by Streamlit for 1 hour to avoid excessive connection creation.
    """
    # Generate a unique connection key
    conn_key = f"{DB_SERVER}_{DB_NAME}_{DB_PORT}"

    # Check if a valid connection exists in the pool
    if conn_key in CONNECTION_POOL and CONNECTION_POOL[conn_key] is not None:
        # Check if the connection is still active
        try:
            # Check if connection has been inactive too long
            if conn_key in LAST_ACTIVITY:
                elapsed = (datetime.now() - LAST_ACTIVITY[conn_key]).total_seconds()
                if elapsed > CONNECTION_TIMEOUT:
                    # Close old connection and create a new one
                    try:
                        CONNECTION_POOL[conn_key].close()
                    except:
                        pass
                    CONNECTION_POOL[conn_key] = None
                    st.sidebar.info(f"Database connection timeout ({elapsed:.0f}s). Reconnecting...")
                elif elapsed > CONNECTION_KEEPALIVE:
                    # Send a keepalive ping
                    try:
                        cursor = CONNECTION_POOL[conn_key].cursor()
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                        cursor.close()
                        LAST_ACTIVITY[conn_key] = datetime.now()
                        st.sidebar.info("Database connection refreshed.")
                    except:
                        # Connection is stale, will create a new one
                        try:
                            CONNECTION_POOL[conn_key].close()
                        except:
                            pass
                        CONNECTION_POOL[conn_key] = None
                        st.sidebar.warning("Database connection became stale. Reconnecting...")

            # If the connection is still valid, return it
            if CONNECTION_POOL[conn_key] is not None:
                # Verify connection is still working
                cursor = CONNECTION_POOL[conn_key].cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()

                # Update last activity timestamp
                LAST_ACTIVITY[conn_key] = datetime.now()
                return CONNECTION_POOL[conn_key]
        except Exception:
            # Connection is invalid, close it and create a new one
            try:
                CONNECTION_POOL[conn_key].close()
            except:
                pass
            CONNECTION_POOL[conn_key] = None

    # If we get here, we need to create a new connection
    db_username = None
    db_password = None
    use_entra = False
    entra_domain = None

    secrets_available = hasattr(st, 'secrets')

    # Prioritize Streamlit secrets
    if secrets_available:
        try:
            db_username = st.secrets.get("DB_USERNAME")
            db_password = st.secrets.get("DB_PASSWORD")
            db_server = st.secrets.get("DB_SERVER_NAME", DB_SERVER)
            db_name = st.secrets.get("DB_NAME", DB_NAME)
            db_port = st.secrets.get("DB_PORT", DB_PORT)
            use_entra_str = st.secrets.get("USE_ENTRA_AUTH", "false")
            use_entra = use_entra_str.lower() == "true"
            entra_domain = st.secrets.get("ENTRA_DOMAIN")
        except Exception as e:
            st.warning(f"Could not read DB secrets: {e}. Trying environment variables.")

    # Fallback to Environment Variable
    if db_username is None or db_password is None:
        db_username = os.getenv("DB_USERNAME")
        db_password = os.getenv("DB_PASSWORD")
        db_server = os.getenv("DB_SERVER_NAME", DB_SERVER)
        db_name = os.getenv("DB_NAME", DB_NAME)
        db_port = os.getenv("DB_PORT", DB_PORT)
        use_entra_str = os.getenv("USE_ENTRA_AUTH", "false")
        use_entra = use_entra_str.lower() == "true"
        entra_domain = os.getenv("ENTRA_DOMAIN")

    if not db_username or not db_password:
        st.error(
            "Database credentials (DB_USERNAME, DB_PASSWORD) not found in Streamlit secrets or environment variables."
        )
        st.info(
            "To connect to Azure SQL Database, ensure you have set:\n"
            "- `DB_USERNAME` and `DB_PASSWORD` in `.streamlit/secrets.toml` or as environment variables.\n"
            "- Optionally, `USE_ENTRA_AUTH=true` and `ENTRA_DOMAIN='yourdomain.onmicrosoft.com'` for Entra auth."
        )
        return None

    conn_str = ""
    try:
        # Determine if we should use SQL or Entra authentication
        if use_entra:
            if not entra_domain:
                st.error("Entra authentication enabled but ENTRA_DOMAIN not set")
                return None

            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={db_server},{db_port};"
                f"DATABASE={db_name};"
                f"Authentication=ActiveDirectoryPassword;"
                f"UID={db_username}@{entra_domain};"
                f"PWD={db_password};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
            )
            st.sidebar.caption(f"Attempting Entra Auth to {db_server},{db_port}")
        else:
            # Default to SQL Authentication
            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={db_server},{db_port};"
                f"DATABASE={db_name};"
                f"UID={db_username};"
                f"PWD={db_password};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
            )
            st.sidebar.caption(f"Attempting SQL Auth to {db_server},{db_port}")

        try:
            # Sleep briefly before connection to avoid rapid retry issues
            time.sleep(1)
            connection = pyodbc.connect(conn_str)
            st.sidebar.success("✅ Connected to Azure SQL database.")

            # Store in connection pool and update activity timestamp
            CONNECTION_POOL[conn_key] = connection
            LAST_ACTIVITY[conn_key] = datetime.now()

            return connection
        except pyodbc.Error as e:
            # If first attempt fails, try with older driver
            if "ODBC Driver 18 for SQL Server" in conn_str:
                st.sidebar.warning("Driver 18 failed, trying Driver 17...")
                conn_str = conn_str.replace("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")

                # Sleep briefly before retry
                time.sleep(1)
                connection = pyodbc.connect(conn_str)
                st.sidebar.success("✅ Connected to Azure SQL database with ODBC Driver 17.")

                # Store in connection pool and update activity timestamp
                CONNECTION_POOL[conn_key] = connection
                LAST_ACTIVITY[conn_key] = datetime.now()

                return connection
            else:
                raise e  # Re-raise the exception if all connection attempts failed

    except pyodbc.Error as e:
        st.sidebar.error(f"Database Connection Error: {e}")
        if "08001" in str(e):
            st.sidebar.warning("Cannot reach the server. Check server name, firewall rules, and network connection.")
        elif "28000" in str(e):
            st.sidebar.warning("Login failed. Check username and password.")
        elif "42000" in str(e):
            st.sidebar.warning("Database access error. Check if the database exists and user has permission.")
        else:
            st.sidebar.warning("Please check DB credentials, server address, database name, and firewall rules.")
        return None
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during DB connection: {e}")
        return None


def get_direct_connection(file_path):
    """
    Create a direct connection to a database file.

    Args:
        file_path (str): Path to the database file

    Returns:
        pyodbc.Connection: A database connection or None if connection failed
    """
    try:
        # Determine connection string based on file extension
        if file_path.lower().endswith(".mdb") or file_path.lower().endswith(".accdb"):
            # Access database
            conn_string = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={file_path}"
        elif file_path.lower().endswith(".csv"):
            # CSV file - handle differently, perhaps return a special connector or loader
            return None  # Implement CSV handling separately
        else:
            # Assume SQL Server compatible file
            conn_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};AttachDbFilename={file_path};Database=tempdb;Trusted_Connection=Yes"

        conn = pyodbc.connect(conn_string)
        return conn
    except Exception as e:
        print(f"Direct database connection error: {e}")
        return None


def close_all_connections():
    """
    Closes all database connections in the pool.
    This function should be called when the application is shutting down.
    """
    for conn_key, conn in CONNECTION_POOL.items():
        if conn is not None:
            try:
                conn.close()
                st.sidebar.info(f"Closed database connection: {conn_key}")
            except Exception as e:
                st.sidebar.warning(f"Error closing connection {conn_key}: {e}")

    # Clear the pool
    CONNECTION_POOL.clear()
    LAST_ACTIVITY.clear()


def ping_connections():
    """
    Sends a keep-alive ping to all connections in the pool.
    This function can be called periodically to prevent connections from timing out.
    """
    for conn_key, conn in CONNECTION_POOL.items():
        if conn is not None:
            try:
                # Check if connection has been inactive too long
                if conn_key in LAST_ACTIVITY:
                    elapsed = (datetime.now() - LAST_ACTIVITY[conn_key]).total_seconds()
                    if elapsed > CONNECTION_KEEPALIVE:
                        # Send a keepalive ping
                        cursor = conn.cursor()
                        cursor.execute("SELECT 1")
                        cursor.fetchone()
                        cursor.close()
                        LAST_ACTIVITY[conn_key] = datetime.now()
                        st.sidebar.info(f"Refreshed connection: {conn_key}")
            except Exception as e:
                st.sidebar.warning(f"Connection {conn_key} is stale: {e}")
                # Mark for recreation on next use
                try:
                    conn.close()
                except:
                    pass
                CONNECTION_POOL[conn_key] = None