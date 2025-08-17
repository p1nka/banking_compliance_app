# --- START OF FILE stubs.py ---
"""
This file provides placeholder/stub functions for missing modules
to allow the data_sync_component.py to run without ImportError.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# --- Stubs for missing 'data_sync' module ---

def sync_data(force=False):
    """Placeholder for data synchronization logic."""
    st.session_state['last_sync_time'] = datetime.now()
    return True

def get_data_stats():
    """Placeholder that returns mock data statistics."""
    if 'last_sync_time' not in st.session_state:
        st.session_state['last_sync_time'] = None

    return {
        'azure': {
            'connection': True,
            'record_count': 15234
        },
        'sqlite': {
            'connection': True,
            'record_count': 15234 if st.session_state['last_sync_time'] else 0,
            'last_sync': st.session_state.get('last_sync_time')
        }
    }

# --- Stubs for missing 'ui.sqlbot_ui' module ---

def get_sqlite_schema_for_bot():
    """Placeholder that returns a mock database schema."""
    return {
        'accounts_data': [
            ('Account_ID', 'TEXT'),
            ('Customer_ID', 'TEXT'),
            ('Account_Type', 'TEXT'),
            ('Current_Balance', 'REAL'),
            ('Date_Last_Cust_Initiated_Activity', 'TEXT'),
            ('Expected_Account_Dormant', 'TEXT')
        ],
        'dormant_flags': [
            ('account_id', 'TEXT'),
            ('flag_instruction', 'TEXT'),
            ('timestamp', 'TEXT')
        ]
    }

def get_recent_sql_history():
    """
    Placeholder that returns mock query history.
    FIX: Ensure column names match the column_config in the UI.
    """
    history_data = [
        {
            "ID": 1,
            "Natural Language Query": "count of dormant accounts",
            "SQL Query": "SELECT COUNT(*) FROM accounts_data WHERE Expected_Account_Dormant = 'Yes'",
            "Execution Time (s)": 0.12,
            "Results": 1,
            "Timestamp": datetime.now() - timedelta(minutes=5)
        },
        {
            "ID": 2,
            "Natural Language Query": "show me accounts with high balance",
            "SQL Query": "SELECT * FROM accounts_data WHERE Current_Balance > 100000",
            "Execution Time (s)": 0.45,
            "Results": 15,
            "Timestamp": datetime.now() - timedelta(hours=1)
        }
    ]
    return pd.DataFrame(history_data)

def execute_nl_query(nl_query, suggest_sql=False):
    """Placeholder for the Natural Language to SQL execution logic."""
    # Mock SQL generation
    sql_query = f"SELECT * FROM accounts_data WHERE Account_Type = 'Savings' -- Mock query for: '{nl_query}'"

    if suggest_sql:
        return None, sql_query

    # Mock execution result
    mock_data = {
        'Account_ID': ['ACC001', 'ACC005'],
        'Customer_ID': ['CUST001', 'CUST003'],
        'Current_Balance': [5000.0, 12000.0]
    }
    results_df = pd.DataFrame(mock_data)
    return results_df, sql_query