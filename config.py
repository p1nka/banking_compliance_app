# --- START OF FILE config.py ---

# config.py
"""
Configuration settings for the application.
"""

import os

# Try to load dotenv if available, but continue if not installed




# --- Application Configuration ---
APP_NAME = "Banking Compliance Assistant"
APP_VERSION = "1.0.0"
APP_TITLE = os.getenv("APP_TITLE", "Internal Audit Bot")
APP_DESCRIPTION = "AI Powered Internal Audit and Banking Compliance Bot"
APP_SUBTITLE = """
This Bot helps identify dormant accounts and analyze compliance requirements 
according to Central Bank of UAE regulations.
"""

# --- Authentication Configuration ---
# FIX: Added missing authentication variables required by auth.py
APP_USERNAME = os.getenv("APP_USERNAME", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "pass123")


# --- Database Configuration ---
# FIX: Consolidated duplicate definitions and standardized on os.getenv()
DB_SERVER = os.getenv("DB_SERVER", "rahulsalpard.database.windows.net")
DB_NAME = os.getenv("DB_NAME", "ra1a")
DB_PORT = os.getenv("DB_PORT", "1433")
DB_USERNAME = os.getenv("DB_USER", "rahulsalpard")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Aug@2025")
DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")


# --- AI Model Configuration ---
AI_MODEL_PROVIDER = os.getenv("AI_MODEL_PROVIDER", "groq")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "llama3-70b-8192")
AI_API_KEY = os.getenv("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") # No default for API key is a safer practice


# --- Feature Flags & Limits ---
ENABLE_AI_FEATURES = os.getenv("ENABLE_AI_FEATURES", "true").lower() == "true"
ENABLE_VISUALIZATION = os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true"
MAX_QUERIES_PER_DAY = int(os.getenv("MAX_QUERIES_PER_DAY", "100"))
MAX_AI_TOKENS = int(os.getenv("MAX_AI_TOKENS", "4096"))


# --- Session State Keys ---
# FIX: Consolidated session keys and added the missing one for login status
SESSION_LOGGED_IN = "user_logged_in"
SESSION_APP_DF = "app_df"
SESSION_CHAT_MESSAGES = "chat_messages"
SESSION_DATA_PROCESSED = "data_processed"
SESSION_COLUMN_MAPPING = "column_mapping"


# --- Default Threshold Values ---
# FIX: Consolidated default threshold values
DEFAULT_DORMANT_DAYS = 365  # 1 year
DEFAULT_FREEZE_DAYS = 730   # 2 years
DEFAULT_CBUAE_DATE = "2023-01-01"  # Default cutoff date for CBUAE transfers