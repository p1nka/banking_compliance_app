# config.py
"""
Configuration file for the Banking Compliance App
Contains database and application settings
"""

# Azure SQL Database Configuration
# These are fallback values - primary configuration should be in .streamlit/secrets.toml
DB_SERVER = "rahulsalpard.database.windows.net"  # Cleaned up server name
DB_NAME = "rahulsalpard"
DB_USERNAME = "agent123"
DB_PASSWORD = "Aug@2025"
DB_PORT = 1433

# Application Configuration
APP_TITLE = "Banking Compliance App"
APP_VERSION = "1.0.0"

# Database Connection Settings
CONNECTION_TIMEOUT = 30  # seconds
LOGIN_TIMEOUT = 30  # seconds
POOL_SIZE = 5
MAX_OVERFLOW = 10
POOL_RECYCLE = 300  # seconds

# Query Limits
DEFAULT_QUERY_LIMIT = 1000
MAX_QUERY_LIMIT = 10000

# Dormancy Analysis Defaults
DEFAULT_DORMANCY_DAYS = 1095  # 3 years
MIN_BALANCE_THRESHOLD = 0.01

# File Upload Settings
MAX_FILE_SIZE_MB = 200
ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Session State Keys
SESSION_KEYS = {
    'connection_status': 'db_connection_status',
    'last_query': 'last_executed_query',
    'query_history': 'sql_query_history',
    'analysis_results': 'dormant_analysis_results',
    'uploaded_data': 'uploaded_data_cache'
}

# Database Table Names
TABLE_NAMES = {
    'accounts': 'accounts_data',
    'dormant_flags': 'dormant_flags',
    'dormant_ledger': 'dormant_ledger',
    'insight_log': 'insight_log',
    'sql_history': 'sql_query_history',
    'analysis_results': 'analysis_results'
}

# Column Mappings for Data Import/Export
COLUMN_MAPPINGS = {
    'account_id': ['Account_ID', 'AccountID', 'account_id', 'id'],
    'customer_id': ['Customer_ID', 'CustomerID', 'customer_id'],
    'account_type': ['Account_Type', 'AccountType', 'account_type', 'type'],
    'currency': ['Currency', 'currency', 'curr'],
    'balance': ['Current_Balance', 'Balance', 'current_balance', 'balance'],
    'last_activity': ['Date_Last_Cust_Initiated_Activity', 'LastActivity', 'last_activity']
}

# Validation Rules
VALIDATION_RULES = {
    'account_id': {
        'required': True,
        'max_length': 255,
        'pattern': r'^[A-Za-z0-9_-]+$'
    },
    'balance': {
        'type': 'decimal',
        'min_value': 0,
        'max_digits': 18,
        'decimal_places': 2
    },
    'currency': {
        'allowed_values': ['AED', 'USD', 'EUR', 'GBP', 'SAR'],
        'max_length': 3
    }
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Banking Compliance Dashboard',
    'page_icon': '🏦',
    'layout': 'wide',
    'sidebar_state': 'expanded',
    'theme': {
        'primary_color': '#1f77b4',
        'background_color': '#ffffff',
        'secondary_background_color': '#f0f2f6',
        'text_color': '#262730'
    }
}

# Export Configuration
EXPORT_CONFIG = {
    'formats': ['csv', 'excel', 'json'],
    'max_rows': 100000,
    'chunk_size': 10000,
    'default_filename_prefix': 'banking_compliance_export'
}