# config.py
"""
Configuration settings for the application.
"""

import os

# Try to load dotenv if available, but continue if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv package not installed. Environment variables will still be read, but .env file support is disabled.")
    print("To enable .env file support, run: pip install python-dotenv")

# Database configuration
DB_SERVER = os.getenv("DB_SERVER", "agentdb123.database.windows.net")
DB_NAME = os.getenv("DB_NAME", "compliance_db")
DB_PORT = os.getenv("DB_PORT", "1433")
DB_DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 18 for SQL Server}")

# AI Model configuration
AI_MODEL_PROVIDER = os.getenv("AI_MODEL_PROVIDER", "groq")
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME", "llama3.3-70b-versatile")
AI_API_KEY = os.getenv("AI_API_KEY", "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")

# Application configuration
APP_NAME = "Banking Compliance Assistant"
APP_VERSION = "1.0.0"
APP_TITLE = os.getenv("APP_TITLE", "Internal Audit Bot")
APP_DESCRIPTION = "AI Powered Internal Audit and Banking Compliance Bot"
# Application constants

APP_SUBTITLE = """
This Bot helps identify dormant accounts and analyze compliance requirements 
according to Central Bank of UAE regulations.
"""

import streamlit as st
from datetime import datetime

# Feature flags
ENABLE_AI_FEATURES = os.getenv("ENABLE_AI_FEATURES", "true").lower() == "true"
ENABLE_VISUALIZATION = os.getenv("ENABLE_VISUALIZATION", "true").lower() == "true"

# Rate limits
MAX_QUERIES_PER_DAY = int(os.getenv("MAX_QUERIES_PER_DAY", "100"))
MAX_AI_TOKENS = int(os.getenv("MAX_AI_TOKENS", "6000"))
# Application constants
APP_TITLE = "Internal Audit Bot"
APP_SUBTITLE = """
This bot helps identify dormant accounts and analyze compliance requirements 
according to Central Bank of UAE regulations.
"""

# Session state keys
SESSION_APP_DF = "app_df"                   # DataFrame with processed data
SESSION_CHAT_MESSAGES = "chat_messages"      # Chat history
SESSION_DATA_PROCESSED = "data_processed"    # Flag for data processing
SESSION_COLUMN_MAPPING = "column_mapping"    # Mapping of standardized to original column names

# Database configuration
# Try to get from environment variables or use defaults
DB_SERVER = os.environ.get("DB_SERVER", "agentdb123.database.windows.net")
DB_NAME = os.environ.get("DB_NAME", "compliance_db")
DB_PORT = os.environ.get("DB_PORT", "1433")  # Default for SQL Server

# Default threshold values (days)
DEFAULT_DORMANT_DAYS = 365  # 1 year
DEFAULT_FREEZE_DAYS = 730   # 2 years
DEFAULT_CBUAE_DATE = "2023-01-01"  # Default cutoff date for CBUAE transfers