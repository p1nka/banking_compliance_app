# database/__init__.py
"""
Database Package for Banking Compliance App

This package contains modules for database connection,
schema management, and database operations.
"""

# First, import the connection module
from .connection import get_db_connection, ping_connections

# Import the schema module
from .schema import init_db, get_db_schema

# Import essential operations
from .operations import (
    save_to_db,
    save_summary_to_db,
    save_sql_query_to_history,
    get_recent_sql_history,
    log_flag_instructions
)

# Make the imports available at the package level
__all__ = [
    'get_db_connection',
    'ping_connections',
    'init_db',
    'get_db_schema',
    'save_to_db',
    'save_summary_to_db',
    'save_sql_query_to_history',
    'get_recent_sql_history',
    'log_flag_instructions'
]

# Try to import pipelines if available
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


    # Create dummy pipeline objects for backward compatibility
    agent_db = DummyPipeline()
    output_db = DummyPipeline()

    # Add them to __all__
    __all__.extend(['agent_db', 'output_db'])