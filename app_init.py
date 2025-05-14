"""
Application initialization script.
This should be imported at the beginning of your main Streamlit app.
"""
import streamlit as st
import logging
import os
from datetime import datetime
# Import database schema initialization functions
# Using try/except to handle case when the module might not be available during first run
try:
    from database.schema import initialize_uae_compliance_schema, update_existing_schema
except ImportError:
    # Will be initialized later, once the module is available
    pass

# Configure logging for the entire application here
# This should be the ONLY place where basicConfig is called
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Outputs to console
        # You can also add a file handler if needed:
        # logging.FileHandler("logs/app.log")
    ]
)

# Get logger for this module
logger = logging.getLogger(__name__)

def initialize_database():
    """
    Initialize database tables if they don't exist.
    """
    try:
        # Try to import in case it wasn't available at the top
        try:
            from database.schema import update_existing_schema
        except ImportError:
            logger.warning("Could not import database schema module. Skipping database initialization.")
            return False

        # Update existing schema or create if needed
        if update_existing_schema():
            logger.info("Database schema updated successfully")
            return True
        else:
            logger.warning("Database schema update had issues")
            return False
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False

def initialize_app():
    """
    Initialize all application components.
    """
    try:
        logger.info("Starting application initialization")

        # Create logs directory if you want to use file logging
        # if not os.path.exists('logs'):
        #     os.makedirs('logs')
        #     logger.info("Created logs directory")

        # Initialize database (without stopping the app if it fails)
        try:
            db_init_result = initialize_database()
            if db_init_result:
                logger.info("Database initialized successfully")
            else:
                logger.warning("Database initialization was skipped or had issues")
        except Exception as db_error:
            logger.error(f"Database initialization error: {db_error}")

        # Set up session state for storing app-wide variables if needed
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = True

        # Load any required application-wide configurations
        if 'app_config' not in st.session_state:
            st.session_state.app_config = {
                'app_name': 'UAE Banking Compliance Suite',
                'version': '1.0.0',
                'initialized_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        logger.info("Application initialization completed")
        return True

    except Exception as e:
        logger.error(f"Application initialization error: {e}")
        return False

# Run initialization when this module is imported
initialization_result = initialize_app()