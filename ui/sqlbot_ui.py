import streamlit as st
import pandas as pd
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import DB_NAME, DB_SERVER
from database.connection import get_db_connection
from database.schema import get_db_schema
from database.operations import save_sql_query_to_history, get_recent_sql_history
from ai.llm import (
    SQL_GENERATION_PROMPT,
    SQL_EXPLANATION_PROMPT,
    get_fallback_response
)


def render_sqlbot(llm=None):
    """
    Render the SQL Bot UI for natural language to SQL queries.

    Args:
        llm: The LLM model for generating SQL
    """
    st.header("SQL Database Query Bot")

    # Check if LLM is available
    if not llm:
        st.warning("⚠️ AI Assistant not available. SQL Bot will run in basic mode.")
        render_manual_sql_interface()
        return

    # Add mode selection for query complexity
    query_complexity = st.radio(
        "SQL Generation Mode:",
        ["Standard", "Advanced"],
        horizontal=True,
        index=0  # Default to Standard
    )

    is_advanced_mode = query_complexity == "Advanced"

    # Add mode selection for execution
    execution_mode = st.radio(
        "Execution Mode:",
        ["Generate & Execute", "Generate Only"],
        horizontal=True,
        key="sql_execution_mode"
    )

    is_generate_only = execution_mode == "Generate Only"

    # Check database connection
    conn = get_db_connection()
    if conn is None:
        st.warning("Cannot use SQL Bot: Database connection failed.")
        return

    # Check if there's a direct connection from file upload
    if "db_connection" in st.session_state and st.session_state["db_connection"]:
        selected_table = st.session_state.get("sql_table_schema", "accounts_data")
        st.info(f"🔗 Connected to database. Primary table: **{selected_table}**")
    else:
        st.info(f"🤖 Connected to database: `{DB_NAME}` on server `{DB_SERVER}`")

    # Get database schema
    schema = get_db_schema()
    if schema:
        schema_text = format_schema_for_display(schema)
        
        with st.expander("Show Database Schema"):
            st.code(schema_text, language='text')

        # SQL Bot UI components
        example_placeholder = "e.g., Show me all dormant accounts with their balances"

        nl_query_sqlbot = st.text_area(
            "Ask a database question:",
            placeholder=example_placeholder,
            height=100,
            key="sql_bot_nl_query_input"
        )

        # Single button that adapts based on selected execution mode
        button_text = "Generate SQL" if is_generate_only else "Generate & Execute SQL"
        generate_button = st.button(button_text, key="sql_bot_action_button")

        if generate_button and nl_query_sqlbot:
            process_sql_query(nl_query_sqlbot, schema_text, llm, conn, is_generate_only, is_advanced_mode)

        # Show generated SQL if available
        if 'generated_sql_sqlbot' in st.session_state and st.session_state.generated_sql_sqlbot:
            st.subheader("Generated SQL Query")
            st.code(st.session_state.generated_sql_sqlbot, language='sql')
            
            if is_generate_only and st.button("Execute Generated Query", key="execute_generated_query"):
                execute_sql_query(st.session_state.generated_sql_sqlbot, conn)

        # Show query history
        show_query_history()
    else:
        st.warning("Could not retrieve database schema. SQL Bot is limited.")
        render_manual_sql_interface()


def render_manual_sql_interface():
    """Render manual SQL input interface when AI is not available."""
    st.subheader("Manual SQL Query")
    
    conn = get_db_connection()
    if conn is None:
        st.error("No database connection available.")
        return

    manual_sql = st.text_area(
        "Enter SQL query:",
        value="SELECT TOP 10 * FROM accounts_data",
        height=150,
        key="manual_sql_input"
    )

    if st.button("Execute Query", key="execute_manual_sql"):
        execute_sql_query(manual_sql, conn)

    show_query_history()


def process_sql_query(nl_query, schema_text, llm, conn, is_generate_only, is_advanced_mode):
    """Process natural language query and generate/execute SQL."""
    try:
        with st.spinner(f"🤖 Converting natural language to SQL..."):
            # Generate SQL using LLM
            sql_query_generated = generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode)
            
            if not sql_query_generated:
                st.error("Failed to generate SQL query.")
                return

            # Store in session state
            st.session_state.last_nl_query_sqlbot = nl_query
            st.session_state.generated_sql_sqlbot = sql_query_generated

            # Display the generated SQL
            st.subheader("Generated SQL Query")
            st.code(sql_query_generated, language='sql')

            # Save to history
            save_query_to_history(nl_query, sql_query_generated)

            # Execute SQL if in execution mode
            if not is_generate_only:
                execute_sql_query(sql_query_generated, conn)

    except Exception as e:
        st.error(f"Error processing query: {e}")


def generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode=False):
    """Generate SQL from natural language using LLM."""
    try:
        # Use standard prompt (we removed the advanced one to avoid complexity)
        prompt_template = SQL_GENERATION_PROMPT
        
        nl_to_sql_prompt = PromptTemplate.from_template(prompt_template)
        nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

        sql_query_raw = nl_to_sql_chain.invoke({
            "schema": schema_text,
            "question": nl_query.strip()
        })

        # Clean up the generated SQL
        sql_query_generated = clean_sql_query(sql_query_raw)

        # Validate the generated query
        if not sql_query_generated or not sql_query_generated.lower().strip().startswith("select"):
            st.error("Generated text does not start with SELECT or is empty.")
            return None

        return sql_query_generated

    except Exception as e:
        st.error(f"SQL generation error: {e}")
        return get_fallback_response("sql_generation")


def execute_sql_query(sql_query, conn):
    """Execute SQL query and display results."""
    try:
        cleaned_query = clean_sql_query(sql_query)
        
        st.info(f"🔍 **Executing SQL Query**")
        st.code(cleaned_query, language='sql')

        with st.spinner("⏳ Executing query..."):
            results_df = pd.read_sql(cleaned_query, conn)

        # Display results
        st.subheader("Query Results")
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
            st.info(f"Query returned {len(results_df)} rows.")

            # Add download button
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=f"sql_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # Add to history with results
            add_query_to_history(cleaned_query, results_df)
        else:
            st.info("Query executed successfully but returned no results.")
            add_query_to_history(cleaned_query, results_df)

    except Exception as e:
        st.error(f"Query execution error: {e}")
        add_query_to_history(sql_query, None, error=str(e))


def clean_sql_query(sql_text):
    """Clean and extract valid SQL from raw text."""
    if not sql_text:
        return None

    # Remove markdown code blocks if present
    sql_text = re.sub(r"^```sql\s*|\s*```$", "", sql_text, flags=re.MULTILINE).strip()
    sql_text = re.sub(r"^```\s*|\s*```$", "", sql_text.strip())

    # Look for SELECT statement
    match = re.search(r"SELECT.*", sql_text, re.IGNORECASE | re.DOTALL)
    if match:
        sql_query = match.group(0).strip()
        return sql_query

    return sql_text.strip()


def format_schema_for_display(schema):
    """Format schema dictionary for display."""
    schema_text = "Database Schema:\n"
    for table_name, columns_list in schema.items():
        schema_text += f"\nTable: {table_name}\n"
        schema_text += "Columns:\n"
        for name, dtype in columns_list:
            schema_text += f"  - {name} ({dtype})\n"
    return schema_text


def save_query_to_history(nl_query, sql_query):
    """Save query to database history."""
    try:
        if save_sql_query_to_history(nl_query, sql_query):
            st.success("Query saved to history.")
        else:
            st.warning("Failed to save query to history.")
    except Exception as e:
        st.warning(f"Error saving to history: {e}")


def add_query_to_history(query, results=None, error=None):
    """Add query to session history."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    history_item = {
        'query': query,
        'results': results,
        'error': error,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        'ai_generated': True
    }

    st.session_state.query_history.insert(0, history_item)

    # Limit history size
    if len(st.session_state.query_history) > 20:
        st.session_state.query_history = st.session_state.query_history[:20]


def show_query_history():
    """Display query history."""
    if 'query_history' not in st.session_state or not st.session_state.query_history:
        st.info("No query history available.")
        return

    st.subheader("Query History")
    
    for i, item in enumerate(st.session_state.query_history[:5]):  # Show last 5
        with st.expander(f"Query {i+1} - {item['timestamp']}"):
            st.code(item['query'], language='sql')
            
            if item['error']:
                st.error(f"Error: {item['error']}")
            elif item['results'] is not None and not item['results'].empty:
                st.dataframe(item['results'].head())
                st.caption(f"Returned {len(item['results'])} rows")
            else:
                st.info("No results returned")

    # Button to clear history
    if st.button("Clear History"):
        st.session_state.query_history = []
        st.rerun()


# Quick examples for users
def show_example_queries():
    """Show example queries that users can try."""
    st.subheader("Example Queries")
    
    examples = [
        "Show me all dormant accounts",
        "What are the account types and their counts?",
        "Find accounts with balance over 10000",
        "Show customers with no recent activity",
        "List accounts by currency"
    ]
    
    for example in examples:
        if st.button(f"Try: {example}", key=f"example_{hash(example)}"):
            st.session_state['auto_query'] = example
            st.rerun()
