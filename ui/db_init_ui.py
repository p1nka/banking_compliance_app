import streamlit as st
from database.connection import ping_connections, close_all_connections
import time

# Import pipelines only if they're available
try:
    from database.pipelines import agent_db, output_db

    PIPELINES_AVAILABLE = True
except ImportError:
    PIPELINES_AVAILABLE = False


    # Create dummy objects to avoid breaking existing code
    class DummyPipeline:
        def __init__(self):
            pass

        def initialize_connection(self):
            return False

        def ensure_connection(self):
            return False

        def close(self):
            pass


    agent_db = DummyPipeline()
    output_db = DummyPipeline()


def initialize_database_pipelines():
    """
    Initialize database connections for agent and output pipelines.
    This should be called early in the application startup.

    If the pipeline module is not available, this function will
    still run but won't actually initialize any pipelines.
    """
    # Skip if pipelines are not available
    if not PIPELINES_AVAILABLE:
        st.sidebar.info("Using legacy database connection methods (pipelines not available)")
        return

    # Try to initialize the pipelines
    try:
        st.sidebar.subheader("Database Connections")

        with st.sidebar.spinner("Initializing database connections..."):
            # Initialize agent database pipeline
            agent_conn_ok = agent_db.initialize_connection()
            if agent_conn_ok:
                st.sidebar.success("✅ Agent database pipeline connected")
            else:
                st.sidebar.error("❌ Agent database pipeline connection failed")

            # Initialize output storage pipeline
            output_conn_ok = output_db.initialize_connection()
            if output_conn_ok:
                st.sidebar.success("✅ Output storage pipeline connected")
            else:
                st.sidebar.error("❌ Output storage pipeline connection failed")
    except Exception as e:
        st.sidebar.error(f"Error initializing database pipelines: {e}")
        st.sidebar.info("Falling back to legacy database connections")


def cleanup_database_connections():
    """
    Clean up database connections before application shutdown.
    This should be called when the application is shutting down.
    """
    try:
        # Skip if pipelines are not available
        if not PIPELINES_AVAILABLE:
            return

        # Close agent pipeline connection
        agent_db.close()

        # Close output pipeline connection
        output_db.close()

        # Close all pooled connections
        close_all_connections()
    except Exception as e:
        print(f"Error cleaning up database connections: {e}")


# Register cleanup handler if in main session
if __name__ == "__main__":
    import atexit

    atexit.register(cleanup_database_connections)