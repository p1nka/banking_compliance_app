import os
import streamlit as st

# Azure SQL Database connection parameters
DB_SERVER = os.getenv("DB_SERVER", "agentdb123.database.windows.net")
DB_NAME = os.getenv("DB_NAME", "compliance_db")
DB_PORT = os.getenv("DB_PORT", 1433)  # Port is often 1433 for Azure SQL

# Default credentials for app login
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "pass123")

# Constants for session state keys
SESSION_LOGGED_IN = "logged_in"
SESSION_APP_DF = "app_df"
SESSION_DATA_PROCESSED = "data_processed"
SESSION_CHAT_MESSAGES = "chat_messages"
SESSION_COLUMN_MAPPING = "column_mapping"

# Default thresholds
DEFAULT_DORMANT_DAYS = 3 * 365  # 3 years
DEFAULT_FREEZE_DAYS = 5 * 365  # 5 years
DEFAULT_CBUAE_DATE = "2020-04-24"  # Example CBUAE date

# List of available plot types for visualization
ALLOWED_PLOTS = ['bar', 'pie', 'histogram', 'box', 'scatter']


# Initialize streamlit session state
def init_session_state():
    """Initialize session state variables if they don't exist."""
    if SESSION_LOGGED_IN not in st.session_state:
        st.session_state[SESSION_LOGGED_IN] = False

    if SESSION_APP_DF not in st.session_state:
        st.session_state[SESSION_APP_DF] = None

    if SESSION_DATA_PROCESSED not in st.session_state:
        st.session_state[SESSION_DATA_PROCESSED] = False

    if SESSION_CHAT_MESSAGES not in st.session_state:
        st.session_state[SESSION_CHAT_MESSAGES] = [{"role": "assistant", "content": "Hi! Please upload data first..."}]

    if SESSION_COLUMN_MAPPING not in st.session_state:
        st.session_state[SESSION_COLUMN_MAPPING] = {}
