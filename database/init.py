import streamlit as st
import time

# Try to import the pipelines - this will work only if the pipelines.py file exists
try:
    from .pipelines import agent_db, output_db

    PIPELINES_AVAILABLE = True
except ImportError:
    PIPELINES_AVAILABLE = False


    # Create dummy objects to avoid breaking existing code if pipelines.py doesn't exist
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
    This is a simplified version that won't cause import errors.
    """
    # Show a message in the sidebar
    st.sidebar.subheader("Database Connections")

    # Check if pipelines are available
    if PIPELINES_AVAILABLE:
        try:
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
            st.sidebar.info("Using default database connections")
    else:
        st.sidebar.info("Using default database connections (pipelines not available)")

    # Add a brief sleep to allow UI to update
    time.sleep(0.1)