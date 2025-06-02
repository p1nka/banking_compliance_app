import streamlit as st
import requests
import pandas as pd
from io import StringIO
import time
from datetime import datetime, timedelta

from config import (
    DB_SERVER, DB_NAME, DB_PORT,
    SESSION_APP_DF, SESSION_DATA_PROCESSED, SESSION_COLUMN_MAPPING,
    DEFAULT_DORMANT_DAYS, DEFAULT_FREEZE_DAYS, DEFAULT_CBUAE_DATE
)

# Try to import data parser, create fallback if not available
try:
    from data.parser import parse_data
except ImportError:
    def parse_data(data_source):
        """Fallback parser function"""
        if isinstance(data_source, pd.DataFrame):
            return data_source
        elif hasattr(data_source, 'name'):  # Uploaded file
            if data_source.name.endswith('.csv'):
                return pd.read_csv(data_source)
            elif data_source.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(data_source)
        elif isinstance(data_source, str):  # CSV string
            return pd.read_csv(StringIO(data_source))
        return pd.DataFrame()

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
            "**Load Data from Azure SQL Database**"
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
        render_compliance_thresholds()

    # Add report date and agent name settings
    render_analysis_settings()


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
                        uploaded_data_source = response.text
                        st.sidebar.success("✅ Fetched! Ready to process.")
                except requests.exceptions.RequestException as e:
                    st.sidebar.error(f"❌ URL Fetch Error: {e}")
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing URL data: {e}")
            else:
                st.sidebar.warning("⚠️ Please enter a valid URL")

    elif upload_method == "**Load Data from Azure SQL Database**":
        uploaded_data_source = handle_sql_upload()

    return uploaded_data_source


def handle_sql_upload():
    """Handle loading data from Azure SQL Database."""
    st.sidebar.subheader("Azure SQL Data Loader")
    
    # Simple table selector
    table_options = ["accounts_data", "dormant_flags", "dormant_ledger", "insight_log"]
    selected_table = st.sidebar.selectbox(
        "Select table to load:",
        table_options,
        key="table_selector"
    )

    # Load button
    if st.sidebar.button("Load from Database", key="load_db_button"):
        return load_data_from_database(selected_table)

    return None


def load_data_from_database(table_name):
    """Load data from database table."""
    with st.spinner(f"⏳ Loading data from {table_name}..."):
        try:
            conn = get_db_connection()
            if conn is None:
                st.sidebar.error("❌ Database connection failed")
                return None

            # Simple query to get all data
            query = f"SELECT TOP 1000 * FROM {table_name}"
            df = pd.read_sql(query, conn)

            if df.empty:
                st.sidebar.warning(f"Table '{table_name}' has no data.")
                return None
            else:
                st.sidebar.success(f"✅ Loaded {len(df)} rows from '{table_name}'")
                
                # Show preview
                with st.sidebar.expander("Preview Data"):
                    st.dataframe(df.head(3))
                
                return df

        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {e}")
            return None


def process_uploaded_data(uploaded_data_source):
    """Process the uploaded data and update session state."""
    # Initialize session state variables if they don't exist
    if SESSION_COLUMN_MAPPING not in st.session_state:
        st.session_state[SESSION_COLUMN_MAPPING] = {}
    
    with st.spinner("⏳ Processing and standardizing data..."):
        progress_text = st.sidebar.empty()
        progress_text.info("Starting data processing...")

        try:
            # Process the data
            if isinstance(uploaded_data_source, pd.DataFrame):
                data_copy = uploaded_data_source.copy()
                progress_text.info("Processing DataFrame...")
            else:
                data_copy = uploaded_data_source
                progress_text.info("Parsing uploaded data...")

            # Parse the data
            progress_text.info("Standardizing data...")
            df_parsed = parse_data(data_copy)
            
        except Exception as e:
            st.sidebar.error(f"Error processing data: {e}")
            progress_text.error("Data processing failed")
            return

    if df_parsed is not None and not df_parsed.empty:
        # Data was successfully parsed
        st.session_state[SESSION_APP_DF] = df_parsed
        st.session_state[SESSION_DATA_PROCESSED] = True

        progress_text.success(f"✓ Processed {len(df_parsed)} rows, {len(df_parsed.columns)} columns")

        # Update chat message if available
        try:
            from config import SESSION_CHAT_MESSAGES
            
            # Safely get column mapping
            column_mapping = st.session_state.get(SESSION_COLUMN_MAPPING, {})
            
            # Create column info safely
            cols_info = []
            for col in df_parsed.columns[:5]:  # Limit to first 5 columns
                orig_name = column_mapping.get(col, col)
                cols_info.append(f"`{orig_name}`")

            if cols_info:
                cols_example = ', '.join(cols_info)
                initial_message = (
                    f"Data ({len(df_parsed)} rows) processed! "
                    f"Columns include: {cols_example}...\n"
                    f"You can now use the analysis modes."
                )
            else:
                initial_message = f"Data ({len(df_parsed)} rows) processed successfully!"

            st.session_state[SESSION_CHAT_MESSAGES] = [
                {"role": "assistant", "content": initial_message}
            ]
        except ImportError:
            # SESSION_CHAT_MESSAGES not available, skip
            pass

        st.sidebar.success("✅ Data processed successfully!")
        st.balloons()

    elif df_parsed is not None and df_parsed.empty:
        st.sidebar.error("Data parsing resulted in empty dataset.")
        st.session_state[SESSION_DATA_PROCESSED] = False
        st.session_state[SESSION_APP_DF] = None
    else:
        st.sidebar.error("❌ Data parsing failed.")
        st.session_state[SESSION_DATA_PROCESSED] = False
        st.session_state[SESSION_APP_DF] = None

    # Clear progress text
    progress_text.empty()
    
    # Rerun to update the UI
    st.rerun()


def render_app_modes():
    """Render the application mode selector in the sidebar."""
    st.sidebar.header("🚀 Analysis Modes")
    st.session_state.app_mode = st.sidebar.selectbox(
        "Select Application Mode",
        [
            "🏦 Dormant Account Analyzer",
            "🔒 Dormant Compliance Analyzer",
            "💬 Chatbot Only",
            "🔍 SQL Bot"
        ],
        key="app_mode_selector"
    )


def render_analysis_settings():
    """Render analysis settings in the sidebar."""
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Report date for dormancy analysis
    report_date = st.sidebar.date_input(
        "Report Date (Dormancy)", 
        st.session_state.get("report_date_for_dormancy", datetime.now().date()),
        key="report_date_input"
    )
    st.session_state.report_date_for_dormancy = report_date

    # Agent name for compliance logging
    agent_name = st.sidebar.text_input(
        "Agent Name (Compliance)", 
        st.session_state.get("agent_name_for_compliance", "SystemAuditor"),
        key="agent_name_input"
    )
    st.session_state.agent_name_for_compliance = agent_name


def render_compliance_thresholds():
    """Render the compliance thresholds in the sidebar."""
    st.sidebar.subheader("⏱️ Compliance Thresholds")

    # General inactivity threshold for flagging accounts as dormant
    general_inactivity_threshold_days = st.sidebar.number_input(
        "Flagging Inactivity Threshold (days)",
        min_value=1,
        value=st.session_state.get("flagging_inactivity_threshold_days", DEFAULT_DORMANT_DAYS),
        step=30,
        key="flag_threshold_days"
    )
    
    # Calculate and display the threshold date
    general_inactivity_threshold_date = datetime.now() - timedelta(days=general_inactivity_threshold_days)
    st.sidebar.caption(f"Flagging Threshold: {general_inactivity_threshold_date.strftime('%Y-%m-%d')}")

    # Freeze threshold for accounts that have been dormant for a long time
    freeze_inactivity_threshold_days = st.sidebar.number_input(
        "Freeze Inactivity Threshold (days)",
        min_value=1,
        value=st.session_state.get("freeze_inactivity_threshold_days", DEFAULT_FREEZE_DAYS),
        step=30,
        key="freeze_threshold_days"
    )
    
    freeze_inactivity_threshold_date = datetime.now() - timedelta(days=freeze_inactivity_threshold_days)
    st.sidebar.caption(f"Freeze Threshold: {freeze_inactivity_threshold_date.strftime('%Y-%m-%d')}")

    # CBUAE transfer cutoff date
    cbuae_cutoff_str = st.sidebar.text_input(
        "CBUAE Transfer Cutoff Date (YYYY-MM-DD)",
        value=st.session_state.get("cbuae_cutoff_date_str", DEFAULT_CBUAE_DATE),
        key="cbuae_cutoff_date_input"
    )

    # Store the threshold dates in session state
    try:
        cbuae_cutoff_date = datetime.strptime(cbuae_cutoff_str, "%Y-%m-%d")
        st.sidebar.caption(f"CBUAE cutoff: {cbuae_cutoff_date.strftime('%Y-%m-%d')}")

        # Store computed values in session state
        st.session_state["general_threshold_date"] = general_inactivity_threshold_date
        st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date
        st.session_state["cbuae_cutoff_date"] = cbuae_cutoff_date
        st.session_state["flagging_inactivity_threshold_days"] = general_inactivity_threshold_days
        st.session_state["freeze_inactivity_threshold_days"] = freeze_inactivity_threshold_days
        st.session_state["cbuae_cutoff_date_str"] = cbuae_cutoff_str

    except ValueError:
        st.sidebar.error("Invalid CBUAE cutoff date format. Using defaults.")
        # Store only the valid thresholds
        st.session_state["general_threshold_date"] = general_inactivity_threshold_date
        st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date
        st.session_state["cbuae_cutoff_date"] = None
        st.session_state["flagging_inactivity_threshold_days"] = general_inactivity_threshold_days
        st.session_state["freeze_inactivity_threshold_days"] = freeze_inactivity_threshold_days


def show_database_status():
    """Show database connection status in sidebar."""
    st.sidebar.subheader("🔌 Database Status")
    
    if st.sidebar.button("Test Connection"):
        conn = get_db_connection()
        if conn:
            try:
                # Test with a simple query
                test_df = pd.read_sql("SELECT 1 as test", conn)
                if not test_df.empty:
                    st.sidebar.success("✅ Database Connected")
                else:
                    st.sidebar.error("❌ Database Query Failed")
            except Exception as e:
                st.sidebar.error(f"❌ Database Error: {e}")
        else:
            st.sidebar.error("❌ Database Connection Failed")


# Add database status to sidebar
def render_sidebar_with_status():
    """Enhanced sidebar with database status."""
    render_sidebar()
    
    # Add database status at the bottom
    st.sidebar.markdown("---")
    show_database_status()


# Helper function to safely get session state values
def safe_get_session_state(key, default=None):
    """Safely get a value from session state with a default."""
    return st.session_state.get(key, default)


# Initialize required session state variables
def initialize_session_state():
    """Initialize required session state variables."""
    if SESSION_COLUMN_MAPPING not in st.session_state:
        st.session_state[SESSION_COLUMN_MAPPING] = {}
    
    if SESSION_APP_DF not in st.session_state:
        st.session_state[SESSION_APP_DF] = None
        
    if SESSION_DATA_PROCESSED not in st.session_state:
        st.session_state[SESSION_DATA_PROCESSED] = False

    # Initialize analysis settings
    if "report_date_for_dormancy" not in st.session_state:
        st.session_state.report_date_for_dormancy = datetime.now().date()
        
    if "agent_name_for_compliance" not in st.session_state:
        st.session_state.agent_name_for_compliance = "SystemAuditor"


# Call initialization when module is imported
initialize_session_state()
