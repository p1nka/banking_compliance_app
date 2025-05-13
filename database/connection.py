# database/connection.py modifications
import os
import pyodbc
import streamlit as st
import time
import re
from config import DB_SERVER, DB_NAME, DB_PORT

@st.cache_resource(ttl="1h")
def get_db_connection():
    """
    Creates and returns a connection to the Azure SQL database using the credentials
    in st.secrets or environment variables, and global DB_SERVER/DB_NAME.
    Returns None if connection fails or credentials are not found.
    """
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
        except Exception as e:
            st.warning(f"Could not read DB secrets: {e}. Trying environment variables.")

    # Fallback to Environment Variable
    if db_username is None or db_password is None:
        db_username = os.getenv("DB_USERNAME")
        db_password = os.getenv("DB_PASSWORD")
        db_server = os.getenv("DB_SERVER_NAME", DB_SERVER)
        db_name = os.getenv("DB_NAME", DB_NAME)
        db_port = os.getenv("DB_PORT", DB_PORT)

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
            connection = pyodbc.connect(conn_str)
            st.sidebar.success("✅ Connected to Azure SQL database.")
            return connection
        except pyodbc.Error as e:
            # If first attempt fails, try with older driver
            if "ODBC Driver 18 for SQL Server" in conn_str:
                st.sidebar.warning("Driver 18 failed, trying Driver 17...")
                conn_str = conn_str.replace("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")
                connection = pyodbc.connect(conn_str)
                st.sidebar.success("✅ Connected to Azure SQL database with ODBC Driver 17.")
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