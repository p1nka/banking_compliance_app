import os

# Application constants
APP_TITLE = "Banking Compliance Analysis Tool"
APP_SUBTITLE = """
This tool helps identify dormant accounts and analyze compliance requirements 
according to banking regulations.
"""

# Session state keys
SESSION_APP_DF = "app_df"                   # DataFrame with processed data
SESSION_CHAT_MESSAGES = "chat_messages"      # Chat history
SESSION_DATA_PROCESSED = "data_processed"    # Flag for data processing
SESSION_COLUMN_MAPPING = "column_mapping"    # Mapping of standardized to original column names

# Database configuration
# Try to get from environment variables or use defaults
DB_SERVER = os.environ.get("DB_SERVER", "agentdb123.database.windows.net")
DB_NAME = os.environ.get("DB_NAME", "banking_compliance")
DB_PORT = os.environ.get("DB_PORT", "1433")  # Default for SQL Server

# Default threshold values (days)
DEFAULT_DORMANT_DAYS = 365  # 1 year
DEFAULT_FREEZE_DAYS = 730   # 2 years
DEFAULT_CBUAE_DATE = "2023-01-01"  # Default cutoff date for CBUAE transfers