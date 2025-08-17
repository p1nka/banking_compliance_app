# --- START OF FILE utils.py ---

# common/utils.py
"""
Utility functions shared across different modules to prevent circular imports.
"""

import re


def clean_sql_query(raw_sql):
    """
    Clean and extract SQL query from raw text.
    Enhanced to better handle complex queries with window functions.

    Args:
        raw_sql (str): The raw SQL query text to clean

    Returns:
        str: A cleaned SQL query ready for execution
    """
    if not raw_sql:
        return ""

    # First, remove any markdown code blocks
    clean_sql = re.sub(r"```sql\s*|\s*```", "", raw_sql, flags=re.IGNORECASE)
    clean_sql = re.sub(r"```.*?\s*|\s*```", "", clean_sql, flags=re.IGNORECASE)

    # Patterns to find the SQL query. Note the capture groups ().
    # The first pattern is broad and does not need a capture group.
    # The subsequent patterns capture the text *after* the marker.
    sql_query_indicators = [
        r"\b(SELECT|WITH)\b[\s\S]+?(?:;|$)",  # A query starting with SELECT or WITH
        r"SQL QUERY:\s*([\s\S]+?)(?:;|$|```)",  # Capture after "SQL QUERY:"
        r"SQL:\s*([\s\S]+?)(?:;|$|```)",  # Capture after "SQL:"
    ]

    extracted_query = clean_sql  # Default to the cleaned text if no specific pattern matches
    for pattern in sql_query_indicators:
        matches = re.search(pattern, clean_sql, re.IGNORECASE | re.DOTALL)
        if matches:
            # FIX: Check if capture groups were used in the pattern.
            # If group 1 exists, it means we captured the content *after* a marker.
            # Otherwise, the whole match (group 0) is the query.
            if matches.lastindex and matches.lastindex >= 1:
                extracted_query = matches.group(1).strip()
            else:
                extracted_query = matches.group(0).strip()
            break  # Stop after the first successful pattern match

    # Remove trailing semicolons as they can cause issues with some drivers
    final_sql = extracted_query.rstrip(';')

    # FIX: Removed the incorrect replacement of escaped single quotes.
    # In SQL, '' is the correct way to escape a single quote inside a string.
    # Replacing it with ' would cause syntax errors.
    # For example, `WHERE name = 'O''Malley'` would become `WHERE name = 'O'Malley'`, which is invalid.
    # clean_sql = clean_sql.replace("''", "'") # This line was incorrect and has been removed.

    return final_sql.strip()


def get_fallback_response(response_type):
    """
    Provides fallback responses when LLM generation fails.

    Args:
        response_type (str): The type of response needed (sql_generation, sql_explanation, etc.)

    Returns:
        str: A fallback response
    """
    fallback_responses = {
        "sql_generation": "SELECT TOP 10 * FROM accounts_data",
        "sql_explanation": """
        This is a basic SELECT query that retrieves data from a table.

        It's a fallback query since there was an error generating or explaining 
        the more complex query you requested. It simply returns the first 10 rows
        from the specified table to give you a preview of the data.
        """,
        "analysis": "Unable to generate analysis due to an error. Please try rephrasing your request."
    }

    return fallback_responses.get(response_type, "Unable to generate response.")


def analyze_nl_query_for_window_patterns(nl_query):
    """
    Analyze natural language query to detect if window functions might be appropriate.

    Args:
        nl_query (str): The natural language query

    Returns:
        dict: Dictionary with detected patterns and suggestions
    """
    query_lower = nl_query.lower()

    suggestions = {}

    # Detect running total patterns
    running_patterns = ["running", "cumulative", "over time", "progressive",
                        "accumulated", "add up", "sum up", "total so far"]
    if any(pattern in query_lower for pattern in running_patterns):
        suggestions["running_total"] = True

    # Detect ranking patterns
    ranking_patterns = ["rank", "top", "bottom", "highest", "lowest",
                        "best performing", "worst performing", "position", "standing"]
    if any(pattern in query_lower for pattern in ranking_patterns):
        suggestions["ranking"] = True

    # Detect moving average patterns
    moving_patterns = ["moving average", "rolling average", "trend", "last n",
                       "previous periods", "over the past", "recent"]
    if any(pattern in query_lower for pattern in moving_patterns):
        suggestions["moving_calculation"] = True

    # Detect comparison to group patterns
    comparison_patterns = ["compared to", "versus", "against", "difference from average",
                           "how does it compare", "deviation", "versus the average"]
    if any(pattern in query_lower for pattern in comparison_patterns):
        suggestions["group_comparison"] = True

    return suggestions