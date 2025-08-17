# --- START OF FILE pipelines.py ---

import pandas as pd
import streamlit as st
from sqlalchemy import text
import json
from .connection import get_db_connection


class BasePipeline:
    def __init__(self):
        self.engine = get_db_connection()
        if self.engine is None:
            st.error(f"{self.__class__.__name__}: Failed to establish database connection.")

    def execute_query(self, query, params=None):
        """Execute a query and return a DataFrame."""
        if not self.engine: return pd.DataFrame()
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            st.error(f"{self.__class__.__name__} Query Error: {e}")
            return pd.DataFrame()

    def execute_non_query(self, query, params=None):
        """Execute a non-query statement (INSERT, UPDATE, DELETE)."""
        if not self.engine: return False
        try:
            with self.engine.connect() as conn:
                with conn.begin():
                    conn.execute(query, params if params is not None else {})
            return True
        except Exception as e:
            st.error(f"{self.__class__.__name__} Non-Query Error: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.engine:
            self.engine.dispose()


class AgentDatabasePipeline(BasePipeline):
    """Pipeline for agent-related database interactions."""

    def get_dormant_accounts(self, days_threshold=1095):
        query = text("""
            SELECT * FROM accounts_data
            WHERE DATEDIFF(day, Date_Last_Cust_Initiated_Activity, GETDATE()) >= :days
            ORDER BY Date_Last_Cust_Initiated_Activity ASC
        """)
        return self.execute_query(query, params={'days': days_threshold})

    def get_compliance_flags(self, account_id=None):
        if account_id:
            query = text("SELECT * FROM dormant_flags WHERE account_id = :acc_id ORDER BY timestamp DESC")
            return self.execute_query(query, params={'acc_id': account_id})
        else:
            return self.execute_query("SELECT * FROM dormant_flags ORDER BY timestamp DESC")

    def get_ledger_entries(self, account_id=None):
        if account_id:
            query = text("SELECT * FROM dormant_ledger WHERE account_id = :acc_id ORDER BY timestamp DESC")
            return self.execute_query(query, params={'acc_id': account_id})
        else:
            return self.execute_query("SELECT * FROM dormant_ledger ORDER BY timestamp DESC")


class OutputStoragePipeline(BasePipeline):
    """Pipeline for storing analysis results and other outputs."""

    def store_analysis_result(self, analysis_type, analysis_name, result_data, summary=None, record_count=0,
                              created_by="system"):
        if isinstance(result_data, pd.DataFrame):
            result_data = result_data.to_json(orient='records')
        elif isinstance(result_data, (dict, list)):
            result_data = json.dumps(result_data)

        query = text("""
            INSERT INTO analysis_results
            (analysis_type, analysis_name, result_data, result_summary, record_count, created_by)
            VALUES (:type, :name, :data, :summary, :count, :creator)
        """)
        params = {
            'type': analysis_type, 'name': analysis_name, 'data': result_data,
            'summary': summary, 'count': record_count, 'creator': created_by
        }
        return self.execute_non_query(query, params)

    def get_analysis_results(self, analysis_type=None, limit=10):
        # FIX: Use parameterization for the TOP clause value to prevent SQL injection.
        # Most DB drivers don't support parameterizing TOP, so we cast to int for safety.
        safe_limit = int(limit)

        if analysis_type:
            query = text(f"""
                SELECT TOP {safe_limit} id, analysis_type, analysis_name, result_summary, record_count, created_by, timestamp
                FROM analysis_results WHERE analysis_type = :type ORDER BY timestamp DESC
            """)
            return self.execute_query(query, params={'type': analysis_type})
        else:
            query = text(f"SELECT TOP {safe_limit} * FROM analysis_results ORDER BY timestamp DESC")
            return self.execute_query(query)


# Create singletons for easy access
agent_db = AgentDatabasePipeline()
output_db = OutputStoragePipeline()