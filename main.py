import sys

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import hashlib
import hmac
import pyodbc

from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()
import ui
from ai.llm import load_llm
import config as AppConfig
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try different import approaches to handle the wakeup_connection issue
try:
    # First try: Import from database.connection directly
    from database.connection import (
        get_db_connection,
        maintain_connection,
        perform_keepalive,
        test_database_connection,
        show_connection_status
    )

    # Create wakeup_connection as an alias for perform_keepalive
    wakeup_connection = perform_keepalive

    st.sidebar.info("✅ Database connection module loaded successfully")

except ImportError as e1:
    st.sidebar.warning(f"⚠️ Direct import failed: {e1}")

    try:
        # Second try: Import from database package
        from database import (
            get_db_connection,
            maintain_connection,
            perform_keepalive,
            test_database_connection,
            show_connection_status
        )

        # Create wakeup_connection as an alias
        wakeup_connection = perform_keepalive

        st.sidebar.info("✅ Database package loaded successfully")

    except ImportError as e2:
        st.sidebar.error(f"❌ Package import failed: {e2}")

        try:
            # Third try: Import using compatibility module
            from database.compatibility import (
                get_db_connection,
                maintain_connection,
                wakeup_connection,
                test_database_connection,
                show_connection_status
            )

            st.sidebar.info("✅ Database compatibility module loaded")

        except ImportError as e3:
            st.sidebar.error(f"❌ Compatibility import failed: {e3}")

# ⚠️ IMPORTANT: Set page config FIRST, before any other imports or operations ⚠️
st.set_page_config(
    page_title="Internal Audit Bot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)
llm = None
config = None

try:
    # 1. Load the LLM instance using your actual function from llm.py
    llm = load_llm()

    # 2. Build the config object the DormantAgent expects, using your real config.py
    # This avoids all mocks and placeholders.
    class LLMConfig:
        def __init__(self, provider, model_name):
            self.provider = provider
            self.model_name = model_name

    class RAGSystemConfig:
        def __init__(self, llm_config):
            self.llm_for_generation = llm_config

    class AgentConfig:
        def __init__(self, rag_system_config):
            self.rag_system = rag_system_config

    # Use the variables directly from your imported config.py
    llm_config = LLMConfig(provider=AppConfig.AI_MODEL_PROVIDER, model_name=AppConfig.AI_MODEL_NAME)
    rag_system_config = RAGSystemConfig(llm_config=llm_config)
    config = AgentConfig(rag_system_config=rag_system_config)

except Exception as e:
    # The load_llm() function already handles and displays errors in the UI
    llm = None
    config = None

# Authentication functions
def check_password():
    """Returns `True` if the user had the correct password."""

    def validate_credentials(username, password):
        # In a production app, you would check against a secure database
        # This is a simple example with hardcoded credentials - replace with secure implementation
        # The passwords should be stored as salted hashes in a database
        credentials = {
            "admin": generate_hash("admin_password"),
            "analyst": generate_hash("analyst_password"),
            "auditor": generate_hash("auditor_password")
        }

        # Check if username exists and password matches
        if username in credentials:
            stored_hash = credentials[username]
            return hmac.compare_digest(stored_hash, generate_hash(password))
        return False

    def generate_hash(password):
        # This is a simple hashing function - in production use a proper password hashing library
        return hashlib.sha256(password.encode()).hexdigest()

    # Initialize session state for login
    if "authentication_status" not in st.session_state:
        st.session_state["authentication_status"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "logout" not in st.session_state:
        st.session_state["logout"] = False

    # If the user is already authenticated, show the logout button
    if st.session_state["authentication_status"]:
        col1, col2 = st.columns([9, 1])
        with col2:
            if st.button("Logout"):
                st.session_state["authentication_status"] = False
                st.session_state["username"] = ""
                st.session_state["logout"] = True
                # Clear application data on logout for security
                if "db_connection" in st.session_state:
                    try:
                        st.session_state["db_connection"].close()
                    except:
                        pass
                    del st.session_state["db_connection"]

                if "SESSION_APP_DF" in st.session_state:
                    del st.session_state["SESSION_APP_DF"]
                if "SESSION_DATA_PROCESSED" in st.session_state:
                    del st.session_state["SESSION_DATA_PROCESSED"]

                st.rerun()  # Changed from st.experimental_rerun()
        return True

    # If the user is not authenticated, show the login form
    if st.session_state["logout"]:
        st.info("You have been logged out.")
        st.session_state["logout"] = False

    # Create a clean login form
    st.markdown("<h1 style='text-align: center;'>Internal Audit Bot</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Login</h3>", unsafe_allow_html=True)

    # Use columns to center the login form
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                if validate_credentials(username, password):
                    st.session_state["authentication_status"] = True
                    st.session_state["username"] = username
                    st.rerun()  # Changed from st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
                    return False

    # Display additional information
    with col2:
        st.markdown("---")
        st.markdown("#### Demo Credentials")
        st.markdown("""
        - Username: `admin`, Password: `admin_password`
        - Username: `analyst`, Password: `analyst_password`
        - Username: `auditor`, Password: `auditor_password`
        """)
        st.markdown("---")
        st.markdown("*Note: In a production environment, credentials should be stored securely.*")

    return False


def create_default_config():
    """Create a default configuration for agents when config is not available."""
    return {
        'dormant_account_config': {
            'safe_deposit_box': {
                'dormancy_period_years': 3,
                'priority_threshold_balance': 10000
            },
            'investment_account': {
                'dormancy_period_years': 3,
                'maturity_check_enabled': True
            },
            'fixed_deposit': {
                'post_maturity_period_years': 3,
                'auto_renewal_check': True
            },
            'demand_deposit': {
                'dormancy_period_years': 3,
                'minimum_balance_threshold': 100
            },
            'payment_instruments': {
                'unclaimed_period_years': 1,
                'instrument_types': ['bankers_cheque', 'bank_draft', 'cashier_order']
            },
            'cb_transfer': {
                'eligibility_period_years': 5,
                'minimum_transfer_amount': 1000
            },
            'article_3_process': {
                'contact_required': True,
                'waiting_period_months': 3
            },
            'high_value_account': {
                'threshold_amount': 100000,
                'special_handling_required': True
            },
            'transition_detection': {
                'reactivation_detection_enabled': True,
                'activity_monitoring_days': 30
            }
        },
        'compliance_settings': {
            'cbuae_compliance_enabled': True,
            'regulatory_reporting_enabled': True,
            'audit_trail_enabled': True
        },
        'general_settings': {
            'default_currency': 'AED',
            'date_format': '%Y-%m-%d',
            'timezone': 'UTC+4'
        }
    }


# Only import other modules after handling login
if check_password():
    # Only after setting page config and authentication, import other modules
    from config import (
        APP_TITLE, APP_SUBTITLE, SESSION_APP_DF,
        SESSION_DATA_PROCESSED, SESSION_COLUMN_MAPPING
    )

    # Initialize session state variables
    from config import SESSION_COLUMN_MAPPING, SESSION_APP_DF, SESSION_DATA_PROCESSED

    if SESSION_COLUMN_MAPPING not in st.session_state:
        st.session_state[SESSION_COLUMN_MAPPING] = {}

    if SESSION_APP_DF not in st.session_state:
        st.session_state[SESSION_APP_DF] = None

    if SESSION_DATA_PROCESSED not in st.session_state:
        st.session_state[SESSION_DATA_PROCESSED] = False
    # Now import UI modules
    from ui.sidebar import render_sidebar
    from ui.dormant_ui import render_dormant_analyzer
    from ui.compliance_ui import render_compliance_analyzer
    from ui.sqlbot_ui import render_sqlbot
    from ui.chatbot_ui import render_chatbot

    # Import AI model
    from ai.llm import load_llm

    # Display user information
    st.sidebar.markdown(f"**Logged in as:** {st.session_state['username']}")
    st.sidebar.markdown("---")

    # App title
    st.title(APP_TITLE)
    st.markdown(APP_SUBTITLE)

    if "report_date_for_dormancy" not in st.session_state:
        st.session_state.report_date_for_dormancy = datetime.now().date()
        # The UI functions expect a string
    report_date_str = st.session_state.report_date_for_dormancy.strftime("%Y-%m-%d")
    if "agent_name_for_compliance" not in st.session_state:
        st.session_state.agent_name_for_compliance = "SystemAuditor"
    agent_name_input = st.session_state.agent_name_for_compliance

    # dormant_flags_history_df for dormant_ui
    dormant_flags_history_df = pd.DataFrame()  # Default to empty
    flagging_inactivity_days = st.session_state.get("flagging_inactivity_threshold_days", 1095)  # from sidebar
    freeze_inactivity_days = st.session_state.get("freeze_inactivity_threshold_days", 1095)  # from sidebar

    report_dt = datetime.strptime(report_date_str, "%Y-%m-%d")  # report_date_str from sidebar date input

    general_threshold_dt = report_dt - timedelta(days=flagging_inactivity_days)
    freeze_threshold_dt = report_dt - timedelta(days=freeze_inactivity_days)
    maintain_connection()
    # Render the sidebar with upload options
    render_sidebar()

    # AGENT COMPATIBILITY FIX: Check if we have data in db_loaded_data but not in SESSION_APP_DF
    # This helps ensure compatibility between SQL data loading and the analyzers
    if "db_loaded_data" in st.session_state and st.session_state.get(SESSION_DATA_PROCESSED, False) == False:
        try:
            from data.parser import parse_data

            # We have SQL data that hasn't been processed for the analyzers
            st.info("Preparing SQL data for analyzers...")

            # Use the existing parse_data function to prepare the data for the analyzers
            df_parsed = parse_data(st.session_state["db_loaded_data"])
            st.session_state[SESSION_APP_DF] = df_parsed
            st.session_state[SESSION_DATA_PROCESSED] = True

            # Show success message
            st.success("SQL data is now available to analyzers. Please select an analysis mode.")
        except Exception as e:
            st.error(f"Error preparing SQL data for analyzers: {e}")

    # Main page content based on selected mode
    if st.session_state.get(SESSION_DATA_PROCESSED, False):
        # Get the data and current mode
        df = st.session_state.get(SESSION_APP_DF)
        app_mode = st.session_state.get("app_mode", "🏦 Dormant Account Analyzer")

        # Display different UI based on selected mode
        if app_mode == "🏦 Dormant Account Analyzer":
            ui.dormant_ui.render_dormant_analyzer(df, report_date_str, llm, dormant_flags_history_df,
                                                  config)  # NEW - Added config parameter
        elif app_mode == "🔒 Dormant Compliance Analyzer":
            ui.compliance_ui.render_compliance_analyzer(df, agent_name_input, llm)
        elif app_mode == "💬 IA Chat":
            render_chatbot(llm)
        elif app_mode == "🔍 SQL Bot":
            render_sqlbot(llm)

    else:
        st.info(
            "👈 Please upload data using the sidebar options to get started."
        )

        st.markdown("""
            ### Essential Columns
            Your dataset should include the following key columns:
            - `Account_ID`: Unique identifier for the account
            - `Customer_ID`: Customer identifier
            - `Account_Type`: Type of account (Savings, Current, Fixed Deposit, etc.)
            - `Date_Last_Cust_Initiated_Activity`: Date of the last customer-initiated activity
            - `Expected_Account_Dormant`: Whether the account is expected to be dormant (Yes/No)

            ### Additional Columns
            For full functionality, include these recommended columns:
            - Date columns: Account creation date, last communication date
            - Status columns: Current balance, auto-renewal settings
            - Flag columns: Customer address known, active liability accounts
            """)

    # Add a footer
    st.markdown("---")
    st.markdown(f"*Banking Compliance Analysis App • {datetime.now().year}*")
else:
    # If login fails, don't load the rest of the application
    pass