import streamlit as st

# Import configuration and core modules
from config import init_session_state, SESSION_APP_DF, SESSION_DATA_PROCESSED
from auth import enforce_login
from database.schema import init_db
from ai.llm import load_llm

# Import UI components
from ui.sidebar import render_sidebar
from ui.dormant_ui import render_dormant_analyzer
from ui.compliance_ui import render_compliance_analyzer
from ui.sqlbot_ui import render_sqlbot
from ui.chatbot_ui import render_chatbot


def main():
    """Main application entry point."""
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Unified Banking Compliance Solution",
        layout="wide"
    )

    # Initialize session state
    init_session_state()

    # Enforce login
    if not enforce_login():
        return

    # Load the LLM for AI features - handle gracefully if it fails
    try:
        llm = load_llm()
        if llm is None:
            st.sidebar.warning("AI features are limited. The application will run in basic mode.")
    except Exception as e:
        st.sidebar.error(f"Error loading AI model: {e}")
        st.sidebar.warning("AI features are disabled. The application will run in basic mode.")
        llm = None

    # Initialize database schema
    try:
        db_initialized = init_db()
    except Exception as db_error:
        st.sidebar.error(f"Database initialization error: {db_error}")
        db_initialized = False

    # Render sidebar (handles data upload and processing)
    render_sidebar()

    # Determine which app mode to display
    app_mode = st.session_state.get("app_mode", None)

    # Get the current DataFrame if data is processed
    current_df = None
    if st.session_state.get(SESSION_DATA_PROCESSED, False) and st.session_state.get(SESSION_APP_DF) is not None:
        current_df = st.session_state[SESSION_APP_DF].copy()

    # Display app title
    st.title(f"{app_mode}" if app_mode else "Unified Banking Compliance Solution")

    # Display processed data overview
    if current_df is not None:
        st.header("Data Overview")
        if st.checkbox("View Processed Dataset (first 5 rows)", key="view_processed_data_checkbox"):
            from config import SESSION_COLUMN_MAPPING
            display_df = current_df.head().copy()

            # Display with original column names if available
            if SESSION_COLUMN_MAPPING in st.session_state and st.session_state[SESSION_COLUMN_MAPPING]:
                try:
                    # Create a display mapping that only includes columns present in the current small dataframe
                    display_columns_mapping = {
                        std_col: st.session_state[SESSION_COLUMN_MAPPING].get(std_col, std_col)
                        for std_col in display_df.columns
                    }
                    display_df.rename(columns=display_columns_mapping, inplace=True)
                    st.dataframe(display_df)
                    st.caption("Displaying original column names where available for the first 5 rows.")
                except Exception as e:
                    st.error(f"Error applying original column names for display: {e}")
                    st.dataframe(current_df.head())
                    st.caption("Displaying standardized column names for the first 5 rows.")
            else:
                st.dataframe(display_df)
                st.caption("Displaying standardized column names for the first 5 rows.")
        st.divider()

    # Render the appropriate app mode interface
    if current_df is not None and app_mode:
        try:
            if app_mode == "🏦 Dormant Account Analyzer":
                render_dormant_analyzer(current_df, llm)

            elif app_mode == "🔒 Compliance Analyzer":
                render_compliance_analyzer(current_df, llm)

            elif app_mode == "🔍 SQL Bot":
                render_sqlbot(llm)

            elif app_mode == "💬 Chatbot Only":
                render_chatbot(current_df, llm)
        except Exception as ui_error:
            st.error(f"Error in UI rendering: {ui_error}")
            import traceback
            st.error(f"Error details: {traceback.format_exc()}")
            st.warning("Try selecting a different mode or refreshing the page.")

    else:  # No data processed
        display_getting_started_info(db_initialized)


def display_getting_started_info(db_initialized):
    """Display getting started information for new users."""
    st.info("👆 Please upload or load data using the sidebar options and click 'Process' to begin analysis.")
    st.header("Getting Started")
    st.markdown("""
    Welcome to the Unified Banking Compliance Solution.
    This application helps you analyze banking account data for compliance purposes, particularly focusing on dormant accounts.

    **Steps:**
    1.  **Upload Data:** Use the sidebar to upload your account data via CSV, XLSX, JSON, or fetch directly from a URL or an Azure SQL Database table.
    2.  **Process Data:** Click the "Process Uploaded/Fetched Data" button. The app will standardize column names and attempt to save the data to the configured default Azure SQL Database.
    3.  **Select Mode:** Once data is processed, choose an analysis mode from the sidebar:
        *   **Dormant Account Analyzer:** Run pre-defined agents to identify different categories of potentially dormant or high-risk accounts.
        *   **Compliance Analyzer:** Run compliance checks (e.g., contact verification, flagging candidates, ledger review, freeze/transfer candidates).
        *   **SQL Bot:** Query the **default** database (where processed data is saved) using natural language (requires AI Assistant).
        *   **Chatbot Only:** Ask questions or request simple visualizations about the **loaded dataset** using natural language (requires AI Assistant).

    **Configuration:**
    *   Database connection and AI features require credentials stored in `.streamlit/secrets.toml` or set as environment variables (`DB_USERNAME`, `DB_PASSWORD`, `GROQ_API_KEY`, etc.).
    *   The default Azure SQL server and database constants can be overridden via environment variables.
    *   Ensure your Azure SQL server firewall allows connections from the IP address where you are running this application.
    """)

    # Database status
    if db_initialized:
        st.success("✅ Default database connected and initialized successfully.")
    else:
        st.warning("⚠️ Default database connection/initialization failed. Check your database configuration.")

    st.markdown("---")
    st.markdown("Developed as a demonstration of AI-powered compliance tools.")


if __name__ == "__main__":
    main()