import os
import streamlit as st
from config import APP_USERNAME, APP_PASSWORD, SESSION_LOGGED_IN

def login():
    """Handles user login via sidebar."""
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    # Default credentials if not found in secrets or environment variables
    app_user = APP_USERNAME
    app_pass = APP_PASSWORD

    # Prefer secrets over environment variables if secrets are available
    secrets_available = hasattr(st, 'secrets')
    if secrets_available:
        try:
            app_user = st.secrets.get("APP_USERNAME", app_user)
            app_pass = st.secrets.get("APP_PASSWORD", app_pass)
        except Exception as e:
             st.sidebar.warning(f"Could not read APP login secrets: {e}. Using default or env vars.")

    if st.sidebar.button("Login"):
        if username == app_user and password == app_pass:
            st.session_state[SESSION_LOGGED_IN] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")

def show_login_info():
    """Display login information in the sidebar."""
    try:
        secrets_or_env_set = (os.getenv("APP_USERNAME") or (hasattr(st, 'secrets') and st.secrets.get("APP_USERNAME")))
        if not secrets_or_env_set:
             st.sidebar.info("Default login: admin / pass123 (Set APP_USERNAME/APP_PASSWORD in secrets.toml or env vars)")
        else:
             st.sidebar.info("Using custom login from secrets/env vars.")
    except Exception:
         st.sidebar.info("Default login: admin / pass123 (Set APP_USERNAME/APP_PASSWORD in secrets.toml or env vars)")

def enforce_login():
    """Check if user is logged in and handle login process if not."""
    if not st.session_state.get(SESSION_LOGGED_IN, False):
        login()
        show_login_info()
        st.stop()  # Stop execution if not logged in
        return False
    return True