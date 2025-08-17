# --- START OF FILE setup.py ---

"""
Database setup script to ensure the correct schema is in place.
This script should be run once to initialize the database.
"""
import logging
from database.schema import init_db, verify_database_structure

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_database():
    """
    Initialize the database by creating all required tables and verifying the schema.

    Returns:
        bool: True if initialization and verification are successful, False otherwise.
    """
    logger.info("Starting database initialization...")
    
    # The init_db function from schema.py handles everything:
    # connection, table creation, and logging.
    init_success = init_db()
    
    if not init_success:
        logger.error("Database schema initialization failed.")
        return False

    logger.info("Database initialization process completed.")
    
    # Additionally, verify the structure as a final check
    verified, message = verify_database_structure()
    if verified:
        logger.info(f"Database structure verification successful: {message}")
        return True
    else:
        logger.error(f"Database structure verification failed: {message}")
        return False

if __name__ == "__main__":
    # This allows the script to be run from the command line
    if initialize_database():
        print("✅ Database setup complete.")
    else:
        print("❌ Database setup failed. Check logs for details.")