import pyodbc
import pandas as pd
import streamlit as st
from datetime import datetime
import time
import json
from .connection import get_db_connection


# Database Pipeline for Agents
class AgentDatabasePipeline:
    def __init__(self):
        self.conn = None
        self.initialize_connection()

    def initialize_connection(self):
        """Initialize and maintain the database connection."""
        try:
            self.conn = get_db_connection()
            if self.conn is None:
                st.error("Agent DB Pipeline: Failed to establish database connection.")
                return False
            return True
        except Exception as e:
            st.error(f"Agent DB Pipeline Error: {e}")
            return False

    def ensure_connection(self):
        """Ensure the connection is active, reconnect if needed."""
        try:
            if self.conn is None:
                return self.initialize_connection()

            # Test connection with a simple query
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
            except pyodbc.Error:
                # Connection is stale, try to reconnect
                st.warning("Agent DB Pipeline: Reconnecting to database...")
                self.conn = None
                time.sleep(2)  # Give it a moment before reconnecting
                return self.initialize_connection()
        except Exception as e:
            st.error(f"Agent DB Pipeline Connection Error: {e}")
            return False

    def execute_query(self, query, params=None):
        """Execute a query and return the results as a DataFrame."""
        if not self.ensure_connection():
            return pd.DataFrame()

        try:
            # Add a small delay to prevent connection issues
            time.sleep(0.5)
            return pd.read_sql(query, self.conn, params=params)
        except pyodbc.Error as e:
            st.error(f"Agent DB Query Error: {e}")
            return pd.DataFrame()

    def execute_non_query(self, query, params=None):
        """Execute a non-query statement (INSERT, UPDATE, DELETE)."""
        if not self.ensure_connection():
            return False

        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
            cursor.close()
            # Add a small delay to stabilize connection
            time.sleep(0.5)
            return True
        except pyodbc.Error as e:
            st.error(f"Agent DB Non-Query Error: {e}")
            return False

    def get_dormant_accounts(self, days_threshold=1095):  # Default 3 years
        """Get accounts with no activity for a specified number of days."""
        query = """
                SELECT * \
                FROM accounts_data
                WHERE DATEDIFF(day, Date_Last_Cust_Initiated_Activity, GETDATE()) >= %s
                ORDER BY Date_Last_Cust_Initiated_Activity ASC \
                """
        return self.execute_query(query, (days_threshold,))

    def get_compliance_flags(self, account_id=None):
        """Get compliance flags for all accounts or a specific account."""
        if account_id:
            query = "SELECT * FROM dormant_flags WHERE account_id = %s ORDER BY timestamp DESC"
            return self.execute_query(query, (account_id,))
        else:
            query = "SELECT * FROM dormant_flags ORDER BY timestamp DESC"
            return self.execute_query(query)

    def get_ledger_entries(self, account_id=None):
        """Get ledger entries for all accounts or a specific account."""
        if account_id:
            query = "SELECT * FROM dormant_ledger WHERE account_id = %s ORDER BY timestamp DESC"
            return self.execute_query(query, (account_id,))
        else:
            query = "SELECT * FROM dormant_ledger ORDER BY timestamp DESC"
            return self.execute_query(query)

    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                st.error(f"Error closing agent database connection: {e}")


# Pipeline for storing results and outputs
class OutputStoragePipeline:
    def __init__(self):
        self.conn = None
        self.initialize_connection()

    def initialize_connection(self):
        """Initialize and maintain the database connection."""
        try:
            self.conn = get_db_connection()
            if self.conn is None:
                st.error("Output Storage Pipeline: Failed to establish database connection.")
                return False
            return True
        except Exception as e:
            st.error(f"Output Storage Pipeline Error: {e}")
            return False

    def ensure_connection(self):
        """Ensure the connection is active, reconnect if needed."""
        try:
            if self.conn is None:
                return self.initialize_connection()

            # Test connection with a simple query
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                return True
            except pyodbc.Error:
                # Connection is stale, try to reconnect
                st.warning("Output Storage Pipeline: Reconnecting to database...")
                self.conn = None
                time.sleep(2)  # Give it a moment before reconnecting
                return self.initialize_connection()
        except Exception as e:
            st.error(f"Output Storage Pipeline Connection Error: {e}")
            return False

    def store_analysis_result(self, analysis_type, analysis_name, result_data, summary=None, record_count=0,
                              created_by="system"):
        """Store analysis results in the database."""
        if not self.ensure_connection():
            return False

        try:
            # Check if the analysis_results table exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'analysis_results'")
            table_exists = cursor.fetchone() is not None
            cursor.close()

            if not table_exists:
                # Create the table if it doesn't exist
                cursor = self.conn.cursor()
                cursor.execute("""
                               CREATE TABLE analysis_results
                               (
                                   id             INT IDENTITY(1,1) PRIMARY KEY,
                                   analysis_type  NVARCHAR(100) NOT NULL,
                                   analysis_name  NVARCHAR(255) NOT NULL,
                                   result_data    NVARCHAR(MAX) NOT NULL,
                                   result_summary NVARCHAR(MAX),
                                   record_count   INT,
                                   created_by     NVARCHAR(100) DEFAULT 'system',
                                   timestamp      DATETIME2 DEFAULT CURRENT_TIMESTAMP
                               )
                               """)
                cursor.execute("""
                               CREATE INDEX IX_analysis_results_analysis_type ON analysis_results (analysis_type, analysis_name);
                               """)
                cursor.close()
                self.conn.commit()

            # Convert complex data structures to JSON string
            if isinstance(result_data, (dict, list, pd.DataFrame)):
                if isinstance(result_data, pd.DataFrame):
                    result_data = result_data.to_dict(orient='records')
                result_data = json.dumps(result_data)

            query = """
                    INSERT INTO analysis_results
                    (analysis_type, analysis_name, result_data, result_summary, record_count, created_by, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP) \
                    """

            cursor = self.conn.cursor()
            cursor.execute(query, (analysis_type, analysis_name, result_data, summary, record_count, created_by))
            self.conn.commit()
            cursor.close()

            # Add a small delay to stabilize connection
            time.sleep(0.5)
            return True
        except pyodbc.Error as e:
            st.error(f"Error storing analysis result: {e}")
            return False

    def get_analysis_results(self, analysis_type=None, limit=10):
        """Retrieve analysis results from the database."""
        if not self.ensure_connection():
            return pd.DataFrame()

        try:
            # Check if the analysis_results table exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'analysis_results'")
            table_exists = cursor.fetchone() is not None
            cursor.close()

            if not table_exists:
                st.warning("Analysis results table does not exist yet.")
                return pd.DataFrame()

            if analysis_type:
                query = """
                        SELECT TOP %s id, analysis_type, \
                               analysis_name, \
                               result_summary,
                               record_count, \
                               created_by, timestamp
                        FROM analysis_results
                        WHERE analysis_type = %s
                        ORDER BY timestamp DESC \
                        """
                return self.execute_query(query, (limit, analysis_type))
            else:
                query = """
                        SELECT TOP %s id, analysis_type, \
                               analysis_name, \
                               result_summary,
                               record_count, \
                               created_by, timestamp
                        FROM analysis_results
                        ORDER BY timestamp DESC \
                        """
                return self.execute_query(query, (limit,))
        except Exception as e:
            st.error(f"Error retrieving analysis results: {e}")
            return pd.DataFrame()

    def execute_query(self, query, params=None):
        """Execute a query and return the results as a DataFrame."""
        if not self.ensure_connection():
            return pd.DataFrame()

        try:
            # Add a small delay to prevent connection issues
            time.sleep(0.5)
            return pd.read_sql(query, self.conn, params=params)
        except pyodbc.Error as e:
            st.error(f"Output Storage Query Error: {e}")
            return pd.DataFrame()

    def save_insight(self, observation, trend, insight, action):
        """Save analysis insight to the insight_log table."""
        if not self.ensure_connection():
            return False

        try:
            query = """
                    INSERT INTO insight_log (timestamp, observation, trend, insight, action)
                    VALUES (CURRENT_TIMESTAMP, %s, %s, %s, %s) \
                    """

            cursor = self.conn.cursor()
            cursor.execute(query, (
                str(observation)[:8000],
                str(trend)[:8000],
                str(insight)[:8000],
                str(action)[:8000]
            ))
            self.conn.commit()
            cursor.close()

            # Add a small delay to stabilize connection
            time.sleep(0.5)
            return True
        except pyodbc.Error as e:
            st.error(f"Error saving insight to database: {e}")
            return False

    def log_dormant_flag(self, account_id, flag_instruction, flag_reason=None, flag_days=None, flagged_by="system"):
        """Log a dormant flag to the dormant_flags table."""
        if not self.ensure_connection():
            return False

        try:
            query = """
                    INSERT INTO dormant_flags
                    (account_id, flag_instruction, flag_reason, flag_days, flagged_by, timestamp)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP) \
                    """

            cursor = self.conn.cursor()
            cursor.execute(query, (account_id, flag_instruction, flag_reason, flag_days, flagged_by))
            self.conn.commit()
            cursor.close()

            # Add a small delay to stabilize connection
            time.sleep(0.5)
            return True
        except pyodbc.Error as e:
            st.error(f"Error logging dormant flag: {e}")
            return False

    def log_ledger_entry(self, account_id, classification, reason=None, classified_by="system"):
        """Log a dormant ledger entry."""
        if not self.ensure_connection():
            return False

        try:
            query = """
                    INSERT INTO dormant_ledger
                    (account_id, classification, classification_reason, classified_by, timestamp)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP) \
                    """

            cursor = self.conn.cursor()
            cursor.execute(query, (account_id, classification, reason, classified_by))
            self.conn.commit()
            cursor.close()

            # Add a small delay to stabilize connection
            time.sleep(0.5)
            return True
        except pyodbc.Error as e:
            st.error(f"Error logging ledger entry: {e}")
            return False

    def close(self):
        """Close the database connection."""
        if self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                st.error(f"Error closing output storage connection: {e}")


# Create singletons for easy access
agent_db = AgentDatabasePipeline()
output_db = OutputStoragePipeline()