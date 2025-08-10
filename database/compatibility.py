# database/compatibility.py
"""
Compatibility module to handle function name changes and imports.
This ensures backward compatibility with existing code.
"""

import streamlit as st

# Import all functions with error handling
try:
    from .connection import (
        get_db_connection,
        perform_keepalive,
        maintain_connection,
        test_database_connection,
        ping_connections,
        force_reconnect,
        close_all_connections,
        show_connection_status
    )

    # Create aliases for backward compatibility
    wakeup_connection = perform_keepalive


    def maintain_connection_wrapper():
        """Wrapper for maintain_connection to handle any compatibility issues."""
        try:
            return maintain_connection()
        except Exception as e:
            st.error(f"Connection maintenance error: {e}")
            return None


    # Additional compatibility functions
    def auto_keepalive_widget():
        """Legacy function name - now uses maintain_connection."""
        return maintain_connection_wrapper()


    def ping_connection():
        """Single connection ping - uses perform_keepalive."""
        return perform_keepalive()

except ImportError as e:
    st.error(f"Database connection module import failed: {e}")


    # Create dummy functions if import fails
    def get_db_connection():
        st.error("❌ Database connection not available")
        return None


    def wakeup_connection():
        st.warning("⚠️ Database wakeup not available")
        return None


    def maintain_connection():
        st.warning("⚠️ Connection maintenance not available")
        return None


    def maintain_connection_wrapper():
        return maintain_connection()


    def auto_keepalive_widget():
        st.warning("⚠️ Auto keepalive widget not available")
        return None


    def perform_keepalive():
        st.warning("⚠️ Keepalive not available")
        return None


    def ping_connection():
        return perform_keepalive()


    def ping_connections():
        st.warning("⚠️ Connection ping not available")
        return None


    def test_database_connection():
        return False, "Connection test not available"


    def force_reconnect():
        st.error("❌ Force reconnect not available")
        return None


    def close_all_connections():
        st.warning("⚠️ Close connections not available")
        return None


    def show_connection_status():
        st.error("❌ Connection status not available")
        return None

# Schema functions
try:
    from .schema import init_db, get_db_schema, verify_database_structure
except ImportError as e:
    st.error(f"Database schema module import failed: {e}")


    def init_db():
        st.error("❌ Database initialization not available")
        return False


    def get_db_schema():
        st.error("❌ Schema retrieval not available")
        return {}


    def verify_database_structure():
        return False, "Schema verification not available"

# Operations functions
try:
    from .operations import (
        save_to_db,
        save_summary_to_db,
        save_sql_query_to_history,
        get_recent_sql_history,
        log_flag_instructions
    )
except ImportError as e:
    st.error(f"Database operations module import failed: {e}")


    def save_to_db(df, table_name="accounts_data"):
        st.error("❌ Database save not available")
        return False


    def save_summary_to_db(observation, trend, insight, action):
        st.error("❌ Summary save not available")
        return False


    def save_sql_query_to_history(nl_query, sql_query):
        st.error("❌ Query history save not available")
        return False


    def get_recent_sql_history(limit=10):
        st.error("❌ Query history retrieval not available")
        return None


    def log_flag_instructions(account_ids, flag_instruction, days_threshold=None):
        return False, "Flag logging not available"

# Export all functions
__all__ = [
    # Connection functions (with compatibility aliases)
    'get_db_connection',
    'wakeup_connection',  # Alias for perform_keepalive
    'perform_keepalive',
    'maintain_connection',
    'maintain_connection_wrapper',
    'auto_keepalive_widget',  # Legacy alias
    'ping_connection',
    'ping_connections',
    'test_database_connection',
    'force_reconnect',
    'close_all_connections',
    'show_connection_status',

    # Schema functions
    'init_db',
    'get_db_schema',
    'verify_database_structure',

    # Operations functions
    'save_to_db',
    'save_summary_to_db',
    'save_sql_query_to_history',
    'get_recent_sql_history',
    'log_flag_instructions'
]


def check_database_modules():
    """
    Check which database modules are available and working.
    Returns a status report.
    """
    status = {
        'connection_module': False,
        'schema_module': False,
        'operations_module': False,
        'overall_status': 'unavailable'
    }

    # Test connection module
    try:
        conn = get_db_connection()
        if conn is not None:
            status['connection_module'] = True
    except:
        pass

    # Test schema module
    try:
        schema = get_db_schema()
        if schema:
            status['schema_module'] = True
    except:
        pass

    # Test operations module
    try:
        # Try a simple operation test
        history = get_recent_sql_history(1)
        status['operations_module'] = True  # If no error, module is available
    except:
        pass

    # Determine overall status
    if all([status['connection_module'], status['schema_module'], status['operations_module']]):
        status['overall_status'] = 'fully_available'
    elif any([status['connection_module'], status['schema_module'], status['operations_module']]):
        status['overall_status'] = 'partially_available'
    else:
        status['overall_status'] = 'unavailable'

    return status


def display_database_status():
    """
    Display the current database module status in Streamlit.
    """
    status = check_database_modules()

    st.sidebar.markdown("### 📊 Database Module Status")

    # Connection module
    if status['connection_module']:
        st.sidebar.success("✅ Connection Module: Available")
    else:
        st.sidebar.error("❌ Connection Module: Unavailable")

    # Schema module
    if status['schema_module']:
        st.sidebar.success("✅ Schema Module: Available")
    else:
        st.sidebar.error("❌ Schema Module: Unavailable")

    # Operations module
    if status['operations_module']:
        st.sidebar.success("✅ Operations Module: Available")
    else:
        st.sidebar.error("❌ Operations Module: Unavailable")

    # Overall status
    if status['overall_status'] == 'fully_available':
        st.sidebar.success("🟢 Overall Status: Fully Available")
    elif status['overall_status'] == 'partially_available':
        st.sidebar.warning("🟡 Overall Status: Partially Available")
    else:
        st.sidebar.error("🔴 Overall Status: Unavailable")

    return status


# Add to exports
__all__.extend(['check_database_modules', 'display_database_status'])