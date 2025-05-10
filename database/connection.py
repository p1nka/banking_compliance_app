import os
import pyodbc
import streamlit as st
import time
import re
from config import DB_SERVER, DB_NAME, DB_PORT

@st.cache_resource(ttl="1h")  # Cache connection for 1 hour
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
            use_entra_str = st.secrets.get("USE_ENTRA_AUTH", "false")
            use_entra = use_entra_str.lower() == "true"
            if use_entra:
                 entra_domain = st.secrets.get("ENTRA_DOMAIN")
                 if not entra_domain:
                      st.warning("USE_ENTRA_AUTH is true, but ENTRA_DOMAIN is missing in secrets.toml.")
        except Exception as e:
            st.warning(f"Could not read DB secrets: {e}. Trying environment variables.")

    # Fallback to Environment Variable
    if db_username is None or db_password is None:
        db_username = os.getenv("DB_USERNAME")
        db_password = os.getenv("DB_PASSWORD")
        if not use_entra:  # Only check env var for Entra if not already set by secrets
            use_entra_str = os.getenv("USE_ENTRA_AUTH", "false")
            use_entra = use_entra_str.lower() == "true"
            if use_entra:
                 entra_domain = os.getenv("ENTRA_DOMAIN")
                 if not entra_domain:
                      st.warning("USE_ENTRA_AUTH env var is true, but ENTRA_DOMAIN env var is missing.")

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
        if use_entra:
            if not entra_domain:
                 st.error("Microsoft Entra Authentication requires ENTRA_DOMAIN.")
                 return None
            conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"  # Try newer driver first
                f"SERVER={DB_SERVER};" 
                f"DATABASE={DB_NAME};"
                f"Authentication=ActiveDirectoryPassword;"
                f"UID={db_username}@{entra_domain};"
                f"PWD={db_password};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
            )
            st.sidebar.caption("Attempting Entra Auth")
        else:
             # First attempt with numeric port in SERVER parameter
             conn_str = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"  # Try newer driver first
                f"SERVER={DB_SERVER},{DB_PORT};" 
                f"DATABASE={DB_NAME};"
                f"UID={db_username};"
                f"PWD={db_password};"
                f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
             )
             st.sidebar.caption(f"Attempting SQL Auth to {DB_SERVER},{DB_PORT}")

        try:
            connection = pyodbc.connect(conn_str)
            st.sidebar.success("✅ Connected to default database.")
            return connection
        except pyodbc.Error as e:
            # If first attempt fails, try with older driver
            if "ODBC Driver 18 for SQL Server" in conn_str:
                st.sidebar.warning("Driver 18 failed, trying Driver 17...")
                conn_str = conn_str.replace("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")
                connection = pyodbc.connect(conn_str)
                st.sidebar.success("✅ Connected to default database with ODBC Driver 17.")
                return connection
            # If still failing, try without port specification
            elif "," in conn_str and not use_entra:
                st.sidebar.warning("Connection with port failed, trying without port...")
                conn_str = (
                    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                    f"SERVER={DB_SERVER};"  # Without port
                    f"DATABASE={DB_NAME};"
                    f"UID={db_username};"
                    f"PWD={db_password};"
                    f"Encrypt=yes;TrustServerCertificate=no;Connection Timeout=120;"
                )
                connection = pyodbc.connect(conn_str)
                st.sidebar.success("✅ Connected to default database without port specification.")
                return connection
            else:
                raise e  # Re-raise the exception if all connection attempts failed

    except pyodbc.Error as e:
        st.sidebar.error(f"Default Database Connection Error: {e}")
        # More specific error message based on error code
        if "08001" in str(e):
            st.sidebar.warning("Cannot reach the server. Check server name, firewall rules, and network connection.")
        elif "28000" in str(e):
            st.sidebar.warning("Login failed. Check username and password.")
        elif "42000" in str(e):
            st.sidebar.warning("Database access error. Check if the database exists and user has permission.")
        elif "01000" in str(e) and "TLS" in str(e):
            st.sidebar.warning("SSL/TLS error. Try setting TrustServerCertificate=yes in connection string.")
        else:
            st.sidebar.warning("Please check DB credentials, server address, database name, and firewall rules.")
        return None
    except Exception as e:
        st.sidebar.error(f"An unexpected error occurred during default DB connection: {e}")
        return None

def test_db_connection(connection_string, display_area="sidebar"):
    """
    Tests a database connection string and returns diagnostic information.

    Args:
        connection_string: The connection string to test (password will be masked in output)
        display_area: Where to display messages ("sidebar" or "main")

    Returns:
        True if connection succeeded, False otherwise
    """
    # Mask the password in the connection string for display
    masked_conn_str = re.sub(r"PWD=[^;]*", "PWD=*****", connection_string)

    display_func = st.sidebar if display_area == "sidebar" else st

    try:
        display_func.info(f"Testing connection with: {masked_conn_str}")
        start_time = time.time()
        connection = pyodbc.connect(connection_string, timeout=30)
        end_time = time.time()

        # Test if we can actually execute a simple query
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()

        connection.close()

        display_func.success(f"✅ Connection successful! Time: {end_time - start_time:.2f}s")
        return True
    except pyodbc.Error as e:
        display_func.error(f"❌ Connection failed: {e}")

        # Provide specific guidance based on error codes
        error_str = str(e)
        if "08001" in error_str:
            display_func.warning("Cannot reach the server. Check server name and firewall rules.")
        elif "28000" in error_str or "18456" in error_str:
            display_func.warning("Authentication failed. Check username and password.")
        elif "42000" in error_str:
            display_func.warning("Database access error. Check database name and permissions.")
        elif "01000" in error_str and ("TLS" in error_str or "SSL" in error_str):
            display_func.warning("SSL/TLS error. Try using TrustServerCertificate=yes.")
        elif "IM002" in error_str:
            display_func.warning("Driver not found. Check ODBC driver installation.")

        return False
    except Exception as e:
        display_func.error(f"❌ Unexpected error: {e}")
        return False

def debug_db_connection(server, database, username, password, use_entra=False, entra_domain=None):
    """
    Provides a comprehensive diagnostic of database connection issues by testing multiple
    connection string variations and driver options.

    Returns a dict with test results and recommendations.
    """
    results = {
        "success": False,
        "successful_conn_str": None,
        "tested_variations": [],
        "recommendation": ""
    }

    # Test different ODBC driver versions
    drivers_to_try = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "SQL Server Native Client 11.0",
        "SQL Server"  # Basic fallback
    ]

    # Test with and without port specification
    server_variations = [
        (f"{server},{DB_PORT}", "with port"),
        (server, "without port")
    ]

    # Test with and without TrustServerCertificate
    trust_cert_variations = [
        ("no", "with certificate validation"),
        ("yes", "without certificate validation")
    ]

    for driver in drivers_to_try:
        for server_var, server_desc in server_variations:
            for trust_cert, trust_desc in trust_cert_variations:
                if use_entra:
                    if not entra_domain:
                        continue
                    conn_str = (
                        f"DRIVER={{{driver}}};"
                        f"SERVER={server_var};"
                        f"DATABASE={database};"
                        f"Authentication=ActiveDirectoryPassword;"
                        f"UID={username}@{entra_domain};"
                        f"PWD={password};"
                        f"Encrypt=yes;TrustServerCertificate={trust_cert};Connection Timeout=30;"
                    )
                    test_desc = f"Entra Auth with {driver}, {server_desc}, {trust_desc}"
                else:
                    conn_str = (
                        f"DRIVER={{{driver}}};"
                        f"SERVER={server_var};"
                        f"DATABASE={database};"
                        f"UID={username};"
                        f"PWD={password};"
                        f"Encrypt=yes;TrustServerCertificate={trust_cert};Connection Timeout=30;"
                    )
                    test_desc = f"SQL Auth with {driver}, {server_desc}, {trust_desc}"

                # Test this variation
                try:
                    st.sidebar.text(f"Testing: {test_desc}")
                    start_time = time.time()
                    connection = pyodbc.connect(conn_str, timeout=15)  # Short timeout for testing
                    end_time = time.time()

                    # Try a simple query
                    cursor = connection.cursor()
                    cursor.execute("SELECT 1")
                    cursor.fetchall()
                    cursor.close()

                    connection.close()

                    results["tested_variations"].append({
                        "description": test_desc,
                        "success": True,
                        "time": f"{end_time - start_time:.2f}s"
                    })

                    # If this is our first success, save it
                    if not results["success"]:
                        results["success"] = True
                        results["successful_conn_str"] = conn_str
                        results["recommendation"] = f"Use {test_desc}"

                except Exception as e:
                    results["tested_variations"].append({
                        "description": test_desc,
                        "success": False,
                        "error": str(e)
                    })

    # If all tests failed, provide a comprehensive error analysis
    if not results["success"]:
        error_counts = {}
        for test in results["tested_variations"]:
            error_msg = test["error"]
            if error_msg in error_counts:
                error_counts[error_msg] += 1
            else:
                error_counts[error_msg] = 1

        most_common_error = max(error_counts.items(), key=lambda x: x[1])
        results["recommendation"] = analyze_db_error(most_common_error[0])

    return results

def analyze_db_error(error_msg):
    """Analyzes a database error message and returns recommendations."""
    if "08001" in error_msg:
        return ("Cannot reach the server. Check:\n"
                "1. Server name is correct\n"
                "2. Azure firewall allows your IP\n"
                "3. Network connectivity\n"
                "4. VPN/proxy settings if applicable")
    elif "28000" in error_msg or "18456" in error_msg:
        return ("Authentication failed. Check:\n"
                "1. Username and password are correct\n"
                "2. User exists in the database\n"
                "3. User has permission to access this database\n"
                "4. For Entra auth, verify domain and permissions")
    elif "42000" in error_msg:
        return ("Database access error. Check:\n"
                "1. Database name is correct\n"
                "2. User has permission to access this database\n"
                "3. Database exists on the server")
    elif "01000" in error_msg and ("TLS" in error_msg or "SSL" in error_msg):
        return ("SSL/TLS error. Try:\n"
                "1. Setting TrustServerCertificate=yes\n"
                "2. Updating your ODBC driver\n"
                "3. Installing required certificates")
    elif "IM002" in error_msg:
        return ("ODBC driver not found. Check:\n"
                "1. Install Microsoft ODBC Driver for SQL Server\n"
                "2. Try different driver versions (17, 18)\n"
                "3. Use SQL Server Native Client if available")
    elif "HYT00" in error_msg:
        return ("Connection timeout. Check:\n"
                "1. Server is reachable\n"
                "2. Increase connection timeout value\n"
                "3. Network latency issues")
    else:
        return (f"Unrecognized error: {error_msg}\n"
                "General recommendations:\n"
                "1. Verify server, database, and credential information\n"
                "2. Check firewall rules in Azure\n"
                "3. Test connection from another tool (e.g., SSMS)\n"
                "4. Check logs in Azure Portal")