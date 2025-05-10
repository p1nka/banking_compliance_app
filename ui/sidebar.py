import streamlit as st
import requests
import pandas as pd
from io import StringIO
import time
from datetime import datetime, timedelta
import pyodbc

from config import (
    DB_SERVER, DB_NAME, DB_PORT,
    SESSION_APP_DF, SESSION_DATA_PROCESSED,
    DEFAULT_DORMANT_DAYS, DEFAULT_FREEZE_DAYS, DEFAULT_CBUAE_DATE
)
from data.parser import parse_data
from database.connection import get_db_connection
from database.operations import save_to_db


def render_sidebar():
    """Render the main sidebar with upload options and settings."""
    st.sidebar.header("📤 Data Upload")

    # Upload method selection
    upload_method = st.sidebar.radio(
        "Select upload method:",
        [
            "**Upload File (CSV/XLSX/JSON)**",
            "**Upload via URL**",
            "**Load Data from Azure SQL (into App)**"
        ],
        key="upload_method_radio"
    )

    uploaded_data_source = handle_upload_method(upload_method)

    # Process button
    process_button_disabled = uploaded_data_source is None
    process_clicked = st.sidebar.button(
        "Process Uploaded/Fetched Data",
        key="process_data_button",
        disabled=process_button_disabled
    )

    if process_clicked and uploaded_data_source is not None:
        process_uploaded_data(uploaded_data_source)

    # Display app modes if data is processed
    if st.session_state.get(SESSION_DATA_PROCESSED, False):
        render_app_modes()

        # Display compliance thresholds in sidebar
        render_compliance_thresholds()


def handle_upload_method(upload_method):
    """Handle the selected upload method and return the uploaded data source."""
    uploaded_data_source = None

    if upload_method == "**Upload File (CSV/XLSX/JSON)**":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Account Dataset",
            type=["csv", "xlsx", "xls", "json"],
            key="data_file_uploader"
        )
        if uploaded_file:
            uploaded_data_source = uploaded_file
            st.sidebar.caption(f"Selected: {uploaded_file.name}")

    elif upload_method == "**Upload via URL**":
        url_input = st.sidebar.text_input("Enter CSV file URL:", key="url_input")
        if st.sidebar.button("Fetch Data from URL", key="fetch_url_button"):
            if url_input:
                try:
                    with st.spinner("⏳ Fetching data from URL..."):
                        response = requests.get(url_input, timeout=30)
                        response.raise_for_status()
                        # Assume URL points to a raw CSV file content
                        uploaded_data_source = response.text
                        st.sidebar.success("✅ Fetched! Ready to process.")
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"❌ URL Fetch Error: {e}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing URL data: {e}")
            else:
                st.sidebar.warning("⚠️ Please enter a valid URL")

    elif upload_method == "**Load Data from Azure SQL (into App)**":
        uploaded_data_source = handle_sql_upload()

    return uploaded_data_source


def handle_sql_upload():
    """Handle the SQL upload option and return data if loaded successfully."""
    st.sidebar.subheader("Azure SQL Data Loader")
    st.sidebar.markdown("*(Connect to a specific DB/table to load data into the application's memory.)*")

    use_secrets_for_conn_upload = st.sidebar.checkbox(
        "Use credentials from secrets.toml (for loading)",
        value=True,
        key="use_secrets_checkbox_upload"
    )

    # Default values for manual input fields
    input_db_server_upload = DB_SERVER
    input_db_name_upload = DB_NAME
    input_db_username_upload = ""
    input_db_password_upload = ""
    input_use_entra_upload = False
    input_entra_domain_upload = ""

    # Display manual input fields if use_secrets is False
    if not use_secrets_for_conn_upload:
        st.sidebar.warning("⚠️ Secrets credentials disabled. Please enter manual connection details.")

        input_db_server_upload = st.sidebar.text_input(
            "Azure SQL Server (Loader):",
            value=DB_SERVER,
            key="db_server_input_upload"
        )

        input_db_name_upload = st.sidebar.text_input(
            "Database Name (Loader):",
            value=DB_NAME,
            key="db_name_input_upload"
        )

        input_db_username_upload = st.sidebar.text_input(
            "Username (Loader):",
            key="db_username_input_upload"
        )

        input_db_password_upload = st.sidebar.text_input(
            "Password (Loader):",
            type="password",
            key="db_password_input_upload"
        )

        input_use_entra_upload = st.sidebar.checkbox(
            "Use Microsoft Entra Authentication (Loader)",
            key="use_entra_checkbox_upload"
        )

        if input_use_entra_upload:
            input_entra_domain_upload = st.sidebar.text_input(
                "Microsoft Entra Domain (Loader):",
                key="entra_domain_input_upload",
                placeholder="e.g., yourdomain.onmicrosoft.com"
            )
    else:
        # Using secrets
        st.sidebar.info("Using credentials from secrets.toml/env vars.")
        st.sidebar.info(f"Default Server: {DB_SERVER}")
        st.sidebar.info(f"Default Database: {DB_NAME}")

    # SQL query input
    sql_query_input_upload = st.sidebar.text_area(
        "SQL Query to Load Data:",
        value="SELECT * FROM accounts_data",
        height=100,
        key="azure_sql_query_input_upload"
    )

    # Advanced connection options
    with st.sidebar.expander("Advanced Connection Options"):
        use_port_in_conn = st.checkbox(
            "Include port in connection string",
            value=True,
            key="use_port_in_conn_upload"
        )

        driver_version = st.selectbox(
            "ODBC Driver Version",
            [
                "ODBC Driver 18 for SQL Server",
                "ODBC Driver 17 for SQL Server",
                "SQL Server Native Client 11.0"
            ],
            key="driver_version_upload"
        )

        trust_server_cert = st.checkbox(
            "Trust Server Certificate",
            value=False,
            key="trust_server_cert_upload"
        )

        timeout_seconds = st.number_input(
            "Connection Timeout (seconds)",
            min_value=15,
            max_value=300,
            value=60,
            key="timeout_seconds_upload"
        )

    if st.sidebar.button("Connect & Load Data", key="connect_azure_sql_button_upload"):
        if sql_query_input_upload:
            with st.spinner("⏳ Connecting to Azure SQL Database and loading data..."):
                conn_upload = None
                uploaded_df = None

                try:
                    # Construct connection string
                    conn_str_upload = create_connection_string(
                        use_secrets_for_conn_upload,
                        input_db_server_upload,
                        input_db_name_upload,
                        input_db_username_upload,
                        input_db_password_upload,
                        input_use_entra_upload,
                        input_entra_domain_upload,
                        use_port_in_conn,
                        driver_version,
                        trust_server_cert,
                        timeout_seconds
                    )

                    # Try database connection
                    debug_conn_str = mask_connection_string(conn_str_upload)
                    st.sidebar.info(f"Connecting with: {debug_conn_str}")

                    conn_upload = pyodbc.connect(conn_str_upload)

                    # Run query
                    if conn_upload:
                        try:
                            # First check if the query can be executed without errors
                            test_cursor = conn_upload.cursor()
                            test_cursor.execute(sql_query_input_upload)
                            test_cursor.close()

                            # Now use pd.read_sql_query to get the data as a DataFrame
                            uploaded_df = pd.read_sql_query(sql_query_input_upload, conn_upload)

                            if uploaded_df.empty:
                                st.sidebar.warning("Query returned no results.")
                            else:
                                st.sidebar.success(f"✅ Query successful! Retrieved {len(uploaded_df)} rows.")

                                # Display a preview of the data in the sidebar
                                with st.sidebar.expander("Preview Data"):
                                    st.dataframe(uploaded_df.head(5))

                                # Add an automatic process button
                                if st.sidebar.button("Process this data now", key="auto_process_sql_data"):
                                    process_uploaded_data(uploaded_df)
                                else:
                                    st.sidebar.info(
                                        "Click 'Process Uploaded/Fetched Data' button at the top of the sidebar to use this data."
                                    )

                        except pyodbc.Error as sql_e:
                            st.sidebar.error(f"❌ SQL Query Error: {sql_e}")

                    else:
                        st.sidebar.error("❌ Failed to establish database connection for loading.")

                except pyodbc.Error as e:
                    st.sidebar.error(f"❌ DB Connection Error: {e}")
                    # Provide more specific error guidance
                    if "08001" in str(e):
                        st.sidebar.warning(
                            "Cannot reach the server. Check server name, firewall rules, and network connection."
                        )
                    elif "28000" in str(e):
                        st.sidebar.warning("Login failed. Check username and password.")
                    elif "42000" in str(e):
                        st.sidebar.warning(
                            "Database access error. Check if the database exists and user has permission."
                        )
                    elif "01000" in str(e) and "TLS" in str(e):
                        st.sidebar.warning(
                            "SSL/TLS error. Try enabling 'Trust Server Certificate' in Advanced Connection Options."
                        )

                except ValueError as e:
                    st.sidebar.error(f"❌ Configuration Error: {e}")

                except Exception as e:
                    st.sidebar.error(f"❌ An unexpected error occurred during data loading: {e}")

                finally:
                    # Close connection
                    if conn_upload:
                        try:
                            conn_upload.close()
                            st.sidebar.success("Connection closed successfully.")
                        except Exception as e:
                            st.sidebar.warning(f"Failed to close loading connection: {e}")

                return uploaded_df
        else:
            st.sidebar.warning("⚠️ Please enter an SQL query.")

    return None


def create_connection_string(
        use_secrets, server, database, username, password,
        use_entra, entra_domain, use_port, driver, trust_cert, timeout
):
    """Create a connection string based on the provided parameters."""
    if use_secrets:
        # Get credentials from secrets/env
        db_username_secrets = None
        db_password_secrets = None

        # Try secrets first
        if hasattr(st, 'secrets'):
            db_username_secrets = st.secrets.get("DB_USERNAME")
            db_password_secrets = st.secrets.get("DB_PASSWORD")
            use_entra_secrets_str = st.secrets.get("USE_ENTRA_AUTH", "false")
            use_entra_secrets = use_entra_secrets_str.lower() == "true"
            entra_domain_secrets = st.secrets.get("ENTRA_DOMAIN")

        # Fallback to environment variables
        if not db_username_secrets or not db_password_secrets:
            import os
            db_username_secrets = os.getenv("DB_USERNAME")
            db_password_secrets = os.getenv("DB_PASSWORD")

            if db_username_secrets is None or db_password_secrets is None:
                raise ValueError("Credentials missing in secrets/env vars for loading.")

            use_entra_secrets_str = os.getenv("USE_ENTRA_AUTH", "false")
            use_entra_secrets = use_entra_secrets_str.lower() == "true"
            entra_domain_secrets = os.getenv("ENTRA_DOMAIN")

        # Construct connection string with secrets credentials
        if use_entra_secrets:
            if not entra_domain_secrets:
                raise ValueError("ENTRA_DOMAIN missing")

            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Authentication=ActiveDirectoryPassword;"
                f"UID={db_username_secrets}@{entra_domain_secrets};PWD={db_password_secrets};"
                f"Encrypt=yes;TrustServerCertificate={'yes' if trust_cert else 'no'};Connection Timeout={timeout};"
            )
        else:
            server_with_port = f"{server},{DB_PORT}" if use_port else server
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server_with_port};DATABASE={database};"
                f"UID={db_username_secrets};PWD={db_password_secrets};"
                f"Encrypt=yes;TrustServerCertificate={'yes' if trust_cert else 'no'};Connection Timeout={timeout};"
            )

    else:  # Use manual inputs
        if use_entra:
            if not entra_domain:
                raise ValueError("Manual ENTRA_DOMAIN missing")

            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"Authentication=ActiveDirectoryPassword;"
                f"UID={username}@{entra_domain};PWD={password};"
                f"Encrypt=yes;TrustServerCertificate={'yes' if trust_cert else 'no'};Connection Timeout={timeout};"
            )
        else:
            server_with_port = f"{server},{DB_PORT}" if use_port else server
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server_with_port};DATABASE={database};"
                f"UID={username};PWD={password};"
                f"Encrypt=yes;TrustServerCertificate={'yes' if trust_cert else 'no'};Connection Timeout={timeout};"
            )

    return conn_str


def mask_connection_string(conn_str):
    """Mask password in connection string for display."""
    import re
    return re.sub(r"PWD=[^;]*", "PWD=*****", conn_str)


def process_uploaded_data(uploaded_data_source):
    """Process the uploaded data and update session state."""
    with st.spinner("⏳ Processing and standardizing data..."):
        # Show a progress message
        progress_text = st.sidebar.empty()
        progress_text.info("Starting data processing...")

        # Try to make a copy of the data to avoid reference issues
        try:
            if isinstance(uploaded_data_source, pd.DataFrame):
                data_copy = uploaded_data_source.copy()
                progress_text.info("Made copy of uploaded DataFrame...")
            else:
                data_copy = uploaded_data_source
                progress_text.info("Using uploaded data source directly...")

            # Process the data
            progress_text.info("Parsing data...")
            df_parsed = parse_data(data_copy)  # This function is cached
        except Exception as e:
            st.sidebar.error(f"Error preparing data for processing: {e}")
            import traceback
            st.sidebar.error(f"Traceback: {traceback.format_exc()}")
            df_parsed = None

    if df_parsed is not None and not df_parsed.empty:
        # Data was successfully parsed
        st.session_state[SESSION_APP_DF] = df_parsed
        st.session_state[SESSION_DATA_PROCESSED] = True

        # Update chat message
        from config import SESSION_CHAT_MESSAGES, SESSION_COLUMN_MAPPING
        cols_info = []
        for col in df_parsed.columns:
            orig_name = st.session_state[SESSION_COLUMN_MAPPING].get(col, col)
            cols_info.append(f"`{orig_name}`")

        std_cols_example = ', '.join(cols_info[:min(5, len(cols_info))])
        initial_message = (
            f"Data ({len(df_parsed)} rows) processed! Look for columns like: {std_cols_example}...\n"
            f"You can now use the other modes for analysis."
        )
        st.session_state[SESSION_CHAT_MESSAGES] = [{"role": "assistant", "content": initial_message}]

        # Attempt to save processed data to the default database
        from database.schema import init_db
        db_initialized = init_db()
        if db_initialized:
            with st.spinner("💾 Saving processed data to default Azure SQL DB..."):
                save_success = save_to_db(st.session_state[SESSION_APP_DF])
            if save_success:
                st.sidebar.success(f"✅ Processed data saved to default DB!")
            else:
                st.sidebar.error("Processed data, but failed to save to default DB.")
        else:
            st.sidebar.warning("Skipped saving data to DB because default DB connection/initialization failed.")

        # Show success message
        st.sidebar.success("✅ Data processed successfully!")
        # Use st.balloons for a little celebration
        st.balloons()

    elif df_parsed is not None and df_parsed.empty:
        st.sidebar.error("Source data resulted in an empty dataset after parsing.")
        st.session_state[SESSION_DATA_PROCESSED] = False
        st.session_state[SESSION_APP_DF] = None
    else:
        st.sidebar.error("❌ Data parsing failed. Check the error messages above.")
        st.session_state[SESSION_DATA_PROCESSED] = False
        st.session_state[SESSION_APP_DF] = None

    # Rerun to update the UI based on data_processed state
    st.rerun()


def render_app_modes():
    """Render the application mode selector in the sidebar."""
    st.sidebar.header("🚀 Analysis Modes")
    st.session_state.app_mode = st.sidebar.selectbox(
        "Select Application Mode",
        [
            "🏦 Dormant Account Analyzer",
            "🔒 Compliance Analyzer",
            "🔍 SQL Bot",
            "💬 Chatbot Only"
        ],
        key="app_mode_selector"
    )


def render_compliance_thresholds():
    """Render the compliance thresholds in the sidebar."""
    st.sidebar.subheader("Compliance Thresholds")

    # General inactivity threshold for flagging accounts as dormant
    general_inactivity_threshold_days = st.sidebar.number_input(
        "Flagging Inactivity Threshold (days)",
        min_value=1,
        value=DEFAULT_DORMANT_DAYS,
        step=30,
        key="flag_threshold_days"
    )
    general_inactivity_threshold_date = datetime.now() - timedelta(days=general_inactivity_threshold_days)
    st.sidebar.caption(f"Flagging Threshold: {general_inactivity_threshold_date.strftime('%Y-%m-%d')}")

    # Freeze threshold for accounts that have been dormant for a long time
    freeze_inactivity_threshold_days = st.sidebar.number_input(
        "Freeze Inactivity Threshold (days)",
        min_value=1,
        value=DEFAULT_FREEZE_DAYS,
        step=30,
        key="freeze_threshold_days"
    )
    freeze_inactivity_threshold_date = datetime.now() - timedelta(days=freeze_inactivity_threshold_days)
    st.sidebar.caption(f"Freeze Threshold: {freeze_inactivity_threshold_date.strftime('%Y-%m-%d')}")

    # CBUAE transfer cutoff date
    cbuae_cutoff_str = st.sidebar.text_input(
        "CBUAE Transfer Cutoff Date (YYYY-MM-DD)",
        value=DEFAULT_CBUAE_DATE,
        key="cbuae_cutoff_date_input"  # Changed key to avoid conflict with session state variable
    )

    # Store the threshold dates in session state using the key= parameter
    try:
        cbuae_cutoff_date = datetime.strptime(cbuae_cutoff_str, "%Y-%m-%d")
        st.sidebar.caption(f"Using CBUAE cutoff: {cbuae_cutoff_date.strftime('%Y-%m-%d')}")

        # Instead of directly setting session state variables, use st.session_state["key"] format
        # and use different keys for the input widgets and the computed values
        if "general_threshold_date" not in st.session_state:
            st.session_state["general_threshold_date"] = general_inactivity_threshold_date
        else:
            st.session_state["general_threshold_date"] = general_inactivity_threshold_date

        if "freeze_threshold_date" not in st.session_state:
            st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date
        else:
            st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date

        if "cbuae_cutoff_date" not in st.session_state:
            st.session_state["cbuae_cutoff_date"] = cbuae_cutoff_date
        else:
            st.session_state["cbuae_cutoff_date"] = cbuae_cutoff_date

    except ValueError:
        st.sidebar.error("Invalid CBUAE cutoff date format. Transfer agent will be skipped.")
        # Store only the valid thresholds
        if "general_threshold_date" not in st.session_state:
            st.session_state["general_threshold_date"] = general_inactivity_threshold_date
        else:
            st.session_state["general_threshold_date"] = general_inactivity_threshold_date

        if "freeze_threshold_date" not in st.session_state:
            st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date
        else:
            st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date

        if "cbuae_cutoff_date" not in st.session_state:
            st.session_state["cbuae_cutoff_date"] = None
        else:
            st.session_state["cbuae_cutoff_date"] = None