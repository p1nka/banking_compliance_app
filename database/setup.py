"""
Database setup script to ensure the correct schema is in place.
"""
import logging
from database.connection import get_connection
from database.operations import create_insights_table_if_not_exists

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_database():
    """
    Initialize the database and ensure all required tables exist.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    try:
        # Create the insights_log table if it doesn't exist
        if not create_insights_table_if_not_exists():
            logger.error("Failed to create or check insights_log table")
            return False

        # Check for any legacy columns we don't use anymore
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Check if any deprecated columns exist that we need to handle differently
            cursor.execute("""
                           SELECT COUNT(*)
                           FROM INFORMATION_SCHEMA.COLUMNS
                           WHERE TABLE_NAME = 'insights_log'
                             AND COLUMN_NAME = 'regulation_reference'
                           """)

            has_regulation_reference = cursor.fetchone()[0] > 0

            if has_regulation_reference:
                logger.warning(
                    "Column 'regulation_reference' exists but is not used by the current application version")
        except Exception as schema_e:
            logger.warning(f"Non-critical error checking schema: {schema_e}")
        finally:
            if 'conn' in locals():
                conn.close()

        logger.info("Database initialization completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False