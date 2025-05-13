import os
import streamlit as st
from datetime import datetime

# Azure SQL Database connection parameters - read from secrets or env vars
def get_db_config():
    # Prioritize secrets over env vars if secrets are available
    if hasattr(st, 'secrets'):
        return {
            "server": st.secrets.get("DB_SERVER_NAME", os.getenv("DB_SERVER_NAME", "agentdb123.database.windows.net")),
            "database": st.secrets.get("DB_NAME", os.getenv("DB_NAME", "compliance_db")),
            "port": st.secrets.get("DB_PORT", os.getenv("DB_PORT", 1433)),
            "username": st.secrets.get("DB_USERNAME", os.getenv("DB_USERNAME", "")),
            "password": st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD", ""))
        }
    else:
        return {
            "server": os.getenv("DB_SERVER_NAME", "agentdb123.database.windows.net"),
            "database": os.getenv("DB_NAME", "compliance_db"),
            "port": os.getenv("DB_PORT", 1433),
            "username": os.getenv("DB_USERNAME", ""),
            "password": os.getenv("DB_PASSWORD", "")
        }

# Get DB configuration
db_config = get_db_config()
DB_SERVER = db_config["server"]
DB_NAME = db_config["database"]
DB_PORT = db_config["port"]

# Default credentials for app login
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "pass123")

# Constants for session state keys
SESSION_LOGGED_IN = "logged_in"
SESSION_APP_DF = "app_df"
SESSION_DATA_PROCESSED = "data_processed"
SESSION_CHAT_MESSAGES = "chat_messages"
SESSION_COLUMN_MAPPING = "column_mapping"

# Default thresholds aligned with CBUAE regulation
DEFAULT_DORMANT_DAYS = 3 * 365  # 3 years for initial dormancy (Article 2)
DEFAULT_FREEZE_DAYS = 5 * 365  # 5 years for Central Bank transfer (Article 8)
DEFAULT_CBUAE_DATE = (datetime.now().replace(year=datetime.now().year - 5).strftime('%Y-%m-%d'))  # 5 years ago

# CBUAE Regulation schema - all required fields
CBUAE_SCHEMA_FIELDS = [
    'Account_ID', 'Customer_ID', 'Account_Type', 'Currency', 'Account_Creation_Date',
    'Current_Balance', 'Date_Last_Bank_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
    'FTD_Maturity_Date', 'FTD_Auto_Renewal', 'Date_Last_FTD_Renewal_Claim_Request',
    'Inv_Maturity_Redemption_Date', 'SDB_Charges_Outstanding', 'Date_SDB_Charges_Became_Outstanding',
    'SDB_Tenant_Communication_Received', 'Unclaimed_Item_Trigger_Date', 'Unclaimed_Item_Amount',
    'Date_Last_Cust_Initiated_Activity', 'Bank_Contact_Attempted_Post_Dormancy_Trigger',
    'Date_Last_Bank_Contact_Attempt', 'Customer_Responded_to_Bank_Contact',
    'Date_Claim_Received', 'Claim_Successful', 'Amount_Paid_on_Claim', 'Scenario_Notes',
    'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
    'Customer_Has_Litigation_Regulatory_Reqs', 'Holder_Has_Activity_On_Any_Other_Account',
    'Is_Asset_Only_Customer_Type', 'Expected_Account_Dormant', 'Expected_Requires_Article_3_Process',
    'Expected_Transfer_to_CB_Due'
]

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