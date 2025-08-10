# database/__init__.py
"""
Database Package for Banking Compliance App

This package contains modules for database connection,
schema management, and database operations.
"""

# First, import the connection module with error handling
try:
    from .connection import (
        get_db_connection,
        ping_connections,
        perform_keepalive,
        maintain_connection,
        test_database_connection,
        show_connection_status,
        force_reconnect,
        close_all_connections
    )

    # Add wakeup_connection as an alias for perform_keepalive for backward compatibility
    wakeup_connection = perform_keepalive

except ImportError as e:
    import streamlit as st

    st.error(f"Failed to import connection functions: {e}")


    # Create dummy functions
    def get_db_connection():
        st.error("Database connection not available")
        return None


    def ping_connections():
        st.warning("Database ping not available")


    def perform_keepalive():
        st.warning("Database keepalive not available")


    def wakeup_connection():
        st.warning("Database wakeup not available")


    def maintain_connection():
        st.warning("Database maintenance not available")


    def test_database_connection():
        return False, "Connection module not available"


    def show_connection_status():
        st.error("Cannot show connection status")


    def force_reconnect():
        st.error("Cannot force reconnect")
        return None


    def close_all_connections():
        st.warning("Cannot close connections")

# Import the schema module with error handling
try:
    from .schema import init_db, get_db_schema, verify_database_structure
except ImportError as e:
    import streamlit as st

    st.error(f"Failed to import schema functions: {e}")


    def init_db():
        st.error("Schema initialization not available")
        return False


    def get_db_schema():
        st.error("Schema retrieval not available")
        return {}


    def verify_database_structure():
        return False, "Schema verification not available"

# Import essential operations with error handling
try:
    from .operations import (
        save_to_db,
        save_summary_to_db,
        save_sql_query_to_history,
        get_recent_sql_history,
        log_flag_instructions,
        execute_custom_query,
        get_table_data,
        get_table_count,
        check_database_health
    )
except ImportError as e:
    import streamlit as st

    st.error(f"Failed to import operations functions: {e}")


    def save_to_db(df, table_name="accounts_data"):
        st.error("Database save operation not available")
        return False


    def save_summary_to_db(observation, trend, insight, action):
        st.error("Summary save operation not available")
        return False


    def save_sql_query_to_history(nl_query, sql_query):
        st.error("Query history save not available")
        return False


    def get_recent_sql_history(limit=10):
        st.error("Query history retrieval not available")
        return None


    def log_flag_instructions(account_ids, flag_instruction, days_threshold=None):
        return False, "Flag logging not available"


    def execute_custom_query(query):
        st.error("Custom query execution not available")
        return None


    def get_table_data(table_name, limit=1000):
        st.error("Table data retrieval not available")
        return None


    def get_table_count(table_name):
        st.error("Table count not available")
        return 0


    def check_database_health():
        return {"status": "unavailable", "message": "Database health check not available"}

# Make the imports available at the package level
__all__ = [
    # Connection functions
    'get_db_connection',
    'ping_connections',
    'perform_keepalive',
    'wakeup_connection',  # Added this
    'maintain_connection',
    'test_database_connection',
    'show_connection_status',
    'force_reconnect',
    'close_all_connections',

    # Schema functions
    'init_db',
    'get_db_schema',
    'verify_database_structure',

    # Operations functions
    'save_to_db',
    'save_summary_to_db',
    'save_sql_query_to_history',
    'get_recent_sql_history',
    'log_flag_instructions',
    'execute_custom_query',
    'get_table_data',
    'get_table_count',
    'check_database_health'
]

# Try to import pipelines if available (legacy support)
try:
    from .pipelines import agent_db, output_db

    __all__.extend(['agent_db', 'output_db'])
except ImportError:
    # If pipelines module is not available, create dummy objects
    class DummyPipeline:
        def __getattr__(self, name):
            def _dummy_method(*args, **kwargs):
                import streamlit as st
                st.warning(f"Database pipeline not available. Method {name} called.")
                return None

            return _dummy_method

        def initialize_connection(self):
            return False

        def ensure_connection(self):
            return False

        def close(self):
            pass


    # Create dummy pipeline objects for backward compatibility
    agent_db = DummyPipeline()
    output_db = DummyPipeline()

    # Add them to __all__
    __all__.extend(['agent_db', 'output_db'])


# Database health check function
def check_database_connectivity():
    """
    Comprehensive database connectivity check.
    Returns a dictionary with connection status and details.
    """
    try:
        # Test basic connection
        success, message = test_database_connection()

        if success:
            # Test schema
            schema_success, schema_message = verify_database_structure()

            # Get health info
            health_info = check_database_health()

            return {
                'status': 'connected',
                'connection_message': message,
                'schema_status': schema_success,
                'schema_message': schema_message,
                'health_info': health_info
            }
        else:
            return {
                'status': 'disconnected',
                'connection_message': message,
                'schema_status': False,
                'schema_message': 'Cannot verify schema - no connection',
                'health_info': {'status': 'unhealthy', 'message': 'No connection'}
            }
    except Exception as e:
        return {
            'status': 'error',
            'connection_message': f'Connection check failed: {e}',
            'schema_status': False,
            'schema_message': 'Cannot verify schema - error occurred',
            'health_info': {'status': 'error', 'message': str(e)}
        }


# Add to exports
__all__.append('check_database_connectivity')