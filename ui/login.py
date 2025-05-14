"""
Login module for the UAE Banking Compliance application.
Provides authentication functionality.
"""
import os
import hashlib
import hmac
import json
from datetime import datetime, timedelta

# Define session state keys
SESSION_USER_INFO = "user_info"
SESSION_AUTH_STATUS = "authenticated"
SESSION_AUTH_TIME = "auth_time"
SESSION_LOGIN_ATTEMPTS = "login_attempts"
MAX_LOGIN_ATTEMPTS = 5
LOGIN_TIMEOUT_MINUTES = 15
SESSION_LOGIN_DISABLED_UNTIL = "login_disabled_until"


def get_default_users():
    """Get default users when no external authentication is configured."""
    return {
        "admin": {
            "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
            "full_name": "Administrator",
            "role": "admin",
            "email": "admin@example.com",
            "permissions": ["view_data", "edit_data", "manage_users", "export_data"]
        },
        "user": {
            "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
            "full_name": "Regular User",
            "role": "user",
            "email": "user@example.com",
            "permissions": ["view_data"]
        }
    }


def authenticate_user(username, password):
    """
    Authenticate a user with provided credentials.

    Args:
        username: Username
        password: Plain-text password

    Returns:
        dict: User info if authenticated, None otherwise
    """
    # Get users dictionary
    users = get_default_users()

    # Check if username exists
    if username not in users:
        return None

    # Get user info
    user_info = users[username]

    # Check password
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash != user_info["password_hash"]:
        return None

    return user_info


def render_login_ui():
    """
    Render the login UI and handle authentication.

    Returns:
        bool: True if authenticated, False otherwise
    """
    # Import streamlit only inside the function
    import streamlit as st

    # Initialize session state for authentication
    if SESSION_AUTH_STATUS not in st.session_state:
        st.session_state[SESSION_AUTH_STATUS] = False

    if SESSION_USER_INFO not in st.session_state:
        st.session_state[SESSION_USER_INFO] = None

    if SESSION_LOGIN_ATTEMPTS not in st.session_state:
        st.session_state[SESSION_LOGIN_ATTEMPTS] = 0

    if SESSION_LOGIN_DISABLED_UNTIL not in st.session_state:
        st.session_state[SESSION_LOGIN_DISABLED_UNTIL] = None

    # If already authenticated, return True
    if st.session_state[SESSION_AUTH_STATUS]:
        return True

    # Check if login is disabled due to too many attempts
    login_disabled = False
    if st.session_state[SESSION_LOGIN_DISABLED_UNTIL] is not None:
        now = datetime.now()
        if now < st.session_state[SESSION_LOGIN_DISABLED_UNTIL]:
            remaining_time = st.session_state[SESSION_LOGIN_DISABLED_UNTIL] - now
            minutes = int(remaining_time.total_seconds() / 60)
            seconds = int(remaining_time.total_seconds() % 60)
            st.error(f"⛔ Login temporarily disabled. Try again in {minutes}m {seconds}s.")
            login_disabled = True
        else:
            # Reset if timeout has passed
            st.session_state[SESSION_LOGIN_DISABLED_UNTIL] = None
            st.session_state[SESSION_LOGIN_ATTEMPTS] = 0

    # Display login form if not disabled
    if not login_disabled:
        # Create a centered login form
        st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            background-color: #f8f9fa;
        }
        .login-header {
            text-align: center;
            color: #1E3A8A;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="login-header">UAE Banking Compliance Suite</h2>', unsafe_allow_html=True)

        # Login form
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Log In")

            if submit_button:
                if not username or not password:
                    st.error("Please enter both username and password")
                else:
                    # Authenticate user
                    user_info = authenticate_user(username, password)

                    if user_info:
                        # Authentication successful
                        st.session_state[SESSION_AUTH_STATUS] = True
                        st.session_state[SESSION_USER_INFO] = user_info
                        st.session_state[SESSION_AUTH_TIME] = datetime.now()
                        st.session_state[SESSION_LOGIN_ATTEMPTS] = 0
                        st.success("Login successful! Redirecting...")
                        # Use rerun to refresh the page and apply the authenticated state
                        st.rerun()
                    else:
                        # Authentication failed
                        st.session_state[SESSION_LOGIN_ATTEMPTS] += 1
                        attempts_left = MAX_LOGIN_ATTEMPTS - st.session_state[SESSION_LOGIN_ATTEMPTS]

                        if attempts_left <= 0:
                            # Disable login for a period of time
                            st.session_state[SESSION_LOGIN_DISABLED_UNTIL] = datetime.now() + timedelta(
                                minutes=LOGIN_TIMEOUT_MINUTES)
                            st.error(f"⛔ Too many failed attempts. Login disabled for {LOGIN_TIMEOUT_MINUTES} minutes.")
                        else:
                            st.error(f"❌ Invalid username or password. Attempts left: {attempts_left}")

        # Display default credentials hint
        with st.expander("Need help logging in?"):
            st.info(
                "Default credentials: \n- Username: admin, Password: admin123 \n- Username: user, Password: user123")

        st.markdown('</div>', unsafe_allow_html=True)

    # Return authentication status
    return st.session_state[SESSION_AUTH_STATUS]


def logout_user():
    """Log out the current user by clearing authentication status."""
    import streamlit as st

    if SESSION_AUTH_STATUS in st.session_state:
        st.session_state[SESSION_AUTH_STATUS] = False

    if SESSION_USER_INFO in st.session_state:
        st.session_state[SESSION_USER_INFO] = None


def render_user_info():
    """Render the current user information in the sidebar."""
    import streamlit as st

    if st.session_state.get(SESSION_AUTH_STATUS, False):
        user_info = st.session_state.get(SESSION_USER_INFO, {})

        with st.sidebar.expander("👤 User Information"):
            st.write(f"**Name:** {user_info.get('full_name', 'Unknown')}")
            st.write(f"**Role:** {user_info.get('role', 'Unknown')}")
            st.write(f"**Email:** {user_info.get('email', 'Unknown')}")

            # Add logout button
            if st.button("Logout"):
                logout_user()
                st.rerun()


def check_authentication():
    """
    Check if user is authenticated and redirect to login if not.

    Returns:
        bool: True if authenticated, False otherwise
    """
    import streamlit as st

    # Check if authenticated
    if not st.session_state.get(SESSION_AUTH_STATUS, False):
        return render_login_ui()

    # Check for session timeout (optional)
    auth_time = st.session_state.get(SESSION_AUTH_TIME)
    if auth_time:
        session_duration = datetime.now() - auth_time
        if session_duration > timedelta(hours=8):  # 8-hour session timeout
            st.warning("Your session has expired. Please log in again.")
            logout_user()
            return render_login_ui()

    # User is authenticated
    return True