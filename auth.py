# --- START OF FILE auth.py ---

import os
import streamlit as st
# This import will now work correctly with the fixed config.py
from config import APP_USERNAME, APP_PASSWORD, SESSION_LOGGED_IN


def login():
    """Handles user login via sidebar."""
    st.sidebar.title("🔐 Login")

    # Use secrets if available, otherwise fall back to config defaults
    app_user =os.getenv("APP_USERNAME", APP_USERNAME)
    app_pass = os.getenv("APP_PASSWORD", APP_PASSWORD)

    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")

    if st.sidebar.button("Login", key="login_button"):
        if username == app_user and password == app_pass:
            st.session_state[SESSION_LOGGED_IN] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")


def show_login_info():
    """Display login information in the sidebar."""
    # Check if custom credentials are set either in secrets or environment variables
    secrets_or_env_set = (
            os.getenv("APP_USERNAME") is not None or
            (hasattr(st, 'secrets') and st.secrets.get("APP_USERNAME") is not None)
    )

    if not secrets_or_env_set:
        st.sidebar.info(f"Default login: {APP_USERNAME} / {APP_PASSWORD}")
        st.sidebar.caption("Set APP_USERNAME/APP_PASSWORD in secrets.toml or env vars to override.")
    else:
        st.sidebar.info("Using custom login credentials.")


def enforce_login():
    """
    Checks if a user is logged in. If not, it displays the login form and
    stops the rest of the app from running.
    """
    if not st.session_state.get(SESSION_LOGGED_IN, False):
        login()
        show_login_info()
        st.stop()