import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import DB_NAME, DB_SERVER
from database.connection import get_db_connection
from database.schema import get_db_schema
from database.operations import save_sql_query_to_history, get_recent_sql_history
from ai.llm import SQL_GENERATION_PROMPT, SQL_EXPLANATION_PROMPT


def render_sqlbot(llm):
    """
    Render the SQL Bot UI for natural language to SQL queries.

    Args:
        llm: The LLM model for generating SQL
    """
    st.header("SQL Database Query Bot")

    # Store LLM availability in session state
    st.session_state.llm_available = llm is not None
    if llm is not None:
        st.session_state.llm_instance = llm

    # Explicitly state which database is being queried
    st.info(
        f"🤖 This bot queries the **default database**: `{DB_NAME}` on server `{DB_SERVER}` as configured via secrets/environment variables.")
    st.caption("*(This is separate from the 'Load Data' feature, which brings data into the app's memory.)*")

    # Check prerequisites
    conn = get_db_connection()
    if conn is None:
        st.warning("Cannot use SQL Bot: Default database connection failed.")
        return

    # Fetch schema for the default database
    schema = get_db_schema()
    # Store schema in session state for query explanation
    if schema:
        schema_text = "Database Schema for SQL Bot:\n"
        for table, columns_list in schema.items():
            schema_text += f"Table: {table}\nColumns:\n{chr(10).join([f'- {name} ({dtype})' for name, dtype in columns_list])}\n\n"
        st.session_state.db_schema = schema_text

    # Manual SQL mode if LLM is not available
    if not llm:
        st.warning(
            "AI Assistant (Groq/Langchain) is not available. SQL Bot will run in basic mode with limited functionality.")
        # Provide a simple SQL editor as fallback
        st.subheader("Manual SQL Query")
        manual_sql = st.text_area(
            "Enter SQL query:",
            value="SELECT TOP 10 * FROM accounts_data",
            height=150
        )

        if st.button("Execute Query", key="execute_manual_sql"):
            try:
                with st.spinner("Executing query..."):
                    start_time = time.time()
                    results_df = pd.read_sql(manual_sql, conn)
                    execution_time = round(time.time() - start_time, 3)

                    # Add to query history using new function
                    add_query_to_history(
                        query=manual_sql,
                        results=results_df,
                        explanation="Query executed manually.",
                        error=None,
                        ai_generated=False,
                        execution_time=execution_time
                    )

                st.subheader("Query Results")
                if not results_df.empty:
                    st.dataframe(results_df)
                    st.info(f"Query returned {len(results_df)} rows.")

                    # CSV download button for results
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name=f"sql_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("Query executed successfully but returned no results.")
            except Exception as e:
                st.error(f"Query execution error: {e}")
                # Add failed query to history
                add_query_to_history(
                    query=manual_sql,
                    results=None,
                    explanation=None,
                    error=str(e),
                    ai_generated=False
                )

        # Show query history
        show_query_history()
        return

    # AI-assisted mode
    if schema:
        # Display schema
        with st.expander("Show Database Schema (from default DB)"):
            st.code(schema_text, language='text')

        # SQL Bot UI components
        nl_query_sqlbot = st.text_area(
            "Ask a database question:",
            placeholder="e.g., How many dormant accounts in 'Dubai' branch from the 'accounts_data' table?",
            height=100,
            key="sql_bot_nl_query_input"
        )

        generate_execute_sqlbot = st.button(
            "Generate & Execute SQL Query",
            key="sql_bot_generate_execute_button"
        )

        # Initialize variables
        sql_query_generated = None

        if generate_execute_sqlbot and nl_query_sqlbot:
            # Generate SQL
            try:
                with st.spinner("🤖 Converting natural language to SQL..."):
                    nl_to_sql_prompt = PromptTemplate.from_template(SQL_GENERATION_PROMPT)
                    nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

                    sql_query_raw = nl_to_sql_chain.invoke({
                        "schema": schema_text,
                        "question": nl_query_sqlbot.strip()
                    })

                    # Clean up the generated SQL
                    # Look for SELECT statement
                    match = re.search(r"SELECT.*", sql_query_raw, re.IGNORECASE | re.DOTALL)
                    if match:
                        sql_query_generated = match.group(0).strip()
                    else:
                        # Fallback to standard cleaning if SELECT is not found explicitly by regex
                        sql_query_generated_fallback = re.sub(r"^```sql\s*|\s*```$", "", sql_query_raw,
                                                              flags=re.MULTILINE).strip()
                        st.warning("Could not find SELECT clearly with regex. Using fallback cleaning.")
                        sql_query_generated = sql_query_generated_fallback

                    # Validate the generated query is a SELECT statement
                    if not sql_query_generated or not sql_query_generated.lower().strip().startswith("select"):
                        st.error("Generated text does not start with SELECT or is empty. It cannot be executed.")
                        sql_query_generated = None

            except Exception as e:
                st.error(f"SQL generation error: {e}")
                # Use a fallback SQL
                sql_query_generated = get_fallback_response("sql_generation")

            # Execute SQL if generated successfully
            if sql_query_generated:
                st.subheader("Generated SQL Query")
                st.code(sql_query_generated, language='sql')

                try:
                    with st.spinner("⏳ Executing query..."):
                        # Use pandas to read and display the results
                        start_time = time.time()
                        results_df = pd.read_sql(sql_query_generated, conn)
                        execution_time = round(time.time() - start_time, 3)

                    # Save query to database history
                    try:
                        if save_sql_query_to_history(nl_query_sqlbot, sql_query_generated):
                            st.success("Query saved to history.")
                        else:
                            st.warning("Failed to save query to history.")
                    except Exception as e:
                        st.warning(f"Error saving to history: {e}")

                    # Add to session history for current session
                    add_query_to_history(
                        query=sql_query_generated,
                        results=results_df,
                        explanation=f"Generated from: '{nl_query_sqlbot}'",
                        error=None,
                        ai_generated=True,
                        execution_time=execution_time,
                        nl_query=nl_query_sqlbot
                    )

                    # Display results
                    st.subheader("Query Results")
                    if not results_df.empty:
                        st.dataframe(results_df)
                        st.info(f"Query returned {len(results_df)} rows.")

                        # CSV download button for results
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv_data,
                            file_name=f"sql_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("Query executed successfully but returned no results.")

                except Exception as e:
                    st.error(f"Query execution error: {e}")
                    # Add failed query to history
                    add_query_to_history(
                        query=sql_query_generated,
                        results=None,
                        explanation=f"Generated from: '{nl_query_sqlbot}'",
                        error=str(e),
                        ai_generated=True,
                        nl_query=nl_query_sqlbot
                    )

        # Store in session state for later use
        if 'last_nl_query_sqlbot' not in st.session_state:
            st.session_state.last_nl_query_sqlbot = ""
        if 'generated_sql_sqlbot' not in st.session_state:
            st.session_state.generated_sql_sqlbot = None

        if generate_execute_sqlbot and nl_query_sqlbot:
            st.session_state.last_nl_query_sqlbot = nl_query_sqlbot
            st.session_state.generated_sql_sqlbot = sql_query_generated

        # SQL Explanation button
        if st.session_state.get('generated_sql_sqlbot'):
            if st.button("Explain Generated Query", key="analyze_sql_button_sqlbot"):
                try:
                    with st.spinner("🧠 Analyzing query..."):
                        sql_explanation_prompt = PromptTemplate.from_template(SQL_EXPLANATION_PROMPT)
                        sql_explanation_chain = sql_explanation_prompt | llm | StrOutputParser()

                        explanation = sql_explanation_chain.invoke({
                            "sql_query": st.session_state.generated_sql_sqlbot,
                            "schema": schema_text
                        })
                        st.subheader("Query Analysis")
                        st.markdown(explanation)

                        # Update the most recent query history item with the explanation
                        if 'query_history' in st.session_state and st.session_state.query_history:
                            st.session_state.query_history[0]['explanation'] = explanation

                except Exception as e:
                    st.error(f"Query explanation error: {e}")
                    fallback_explanation = get_fallback_response("sql_explanation")
                    st.warning("Using fallback explanation due to AI error.")
                    st.markdown(fallback_explanation)

                    # Update with fallback explanation
                    if 'query_history' in st.session_state and st.session_state.query_history:
                        st.session_state.query_history[0]['explanation'] = fallback_explanation

        # Show query history
        show_query_history()
    else:
        st.warning("Could not retrieve database schema. SQL Bot is limited.")


def show_query_history():
    """
    Display the history of SQL queries executed in the application.
    Shows both session history and database history.
    """
    st.subheader("Query History")

    # Create tabs for different history views
    history_tabs = st.tabs(["Session History", "Database History"])

    # Tab 1: Session History (using the new approach)
    with history_tabs[0]:
        # Initialize query history in session state if it doesn't exist
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []

        # Check if there's any history to display
        if st.session_state.query_history:
            # Simple filter option
            filter_type = st.selectbox(
                "Filter queries:",
                ["All Queries", "Successful", "Failed", "AI Generated", "Manual"],
                key="filter_session_history"
            )

            # Filter the history based on selection
            filtered_history = st.session_state.query_history

            if filter_type == "Successful":
                filtered_history = [q for q in filtered_history if not q.get('error')]
            elif filter_type == "Failed":
                filtered_history = [q for q in filtered_history if q.get('error')]
            elif filter_type == "AI Generated":
                filtered_history = [q for q in filtered_history if q.get('ai_generated', False)]
            elif filter_type == "Manual":
                filtered_history = [q for q in filtered_history if not q.get('ai_generated', False)]

            # Display each query in the history
            for i, query_item in enumerate(filtered_history):
                # Create a readable title for the expander
                query_title = f"Query #{i + 1}"
                if 'timestamp' in query_item:
                    query_title += f" - {query_item['timestamp']}"

                # Add icons based on query status and type
                if query_item.get('ai_generated', False):
                    query_title = "🤖 " + query_title
                else:
                    query_title = "✏️ " + query_title

                if query_item.get('error'):
                    query_title = "❌ " + query_title
                else:
                    query_title = "✅ " + query_title

                # Create an expander for each query
                with st.expander(query_title):
                    # Create tabs for query details
                    query_tabs = st.tabs(["Query", "Results", "Details"])

                    with query_tabs[0]:  # Query tab
                        # Show NL query if available
                        if 'nl_query' in query_item and query_item['nl_query']:
                            st.markdown("**Natural Language Query:**")
                            st.write(query_item['nl_query'])
                            st.markdown("---")

                        st.markdown("**SQL Query:**")
                        st.code(query_item.get('query', 'No query available'), language='sql')

                        # Add copy button functionality
                        if st.button("📋 Copy SQL", key=f"copy_btn_{i}"):
                            st.code(query_item.get('query', ''))
                            st.success("SQL copied! Use the copy button in the code block above.")

                        # Add re-run button
                        if st.button("🔄 Re-run Query", key=f"rerun_btn_{i}"):
                            st.session_state.sql_query = query_item.get('query', '')

                    with query_tabs[1]:  # Results tab
                        if 'error' in query_item and query_item['error']:
                            st.error(f"Error: {query_item['error']}")
                        elif 'results' in query_item and query_item['results'] is not None:
                            # Check if results is a DataFrame
                            if isinstance(query_item['results'], pd.DataFrame):
                                if not query_item['results'].empty:
                                    st.dataframe(query_item['results'], use_container_width=True)

                                    # Add download button for CSV
                                    csv_data = query_item['results'].to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label="📥 Download as CSV",
                                        data=csv_data,
                                        file_name=f"query_results_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        key=f"download_btn_{i}"
                                    )
                                else:
                                    st.info("Query returned no results.")
                            else:
                                st.write(query_item['results'])
                        else:
                            st.info("No results available.")

                    with query_tabs[2]:  # Details tab
                        st.markdown("**Query Details:**")
                        st.text(f"Timestamp: {query_item.get('timestamp', 'N/A')}")
                        st.text(f"Execution time: {query_item.get('execution_time', 'N/A')} seconds")
                        st.text(f"Generated by: {'AI Assistant' if query_item.get('ai_generated', False) else 'User'}")

                        if 'row_count' in query_item:
                            st.text(f"Rows returned: {query_item.get('row_count', 0)}")

                        # Show explanation if available
                        if 'explanation' in query_item and query_item['explanation']:
                            st.markdown("**Query Explanation:**")
                            st.write(query_item['explanation'])
        else:
            # Display a message if no queries have been run
            st.info("No SQL queries have been executed in this session yet.")

            # Provide a helpful tip
            st.write("**Sample SQL queries you can try:**")
            sample_queries = [
                "SELECT TOP 10 * FROM accounts_data;",
                "SELECT account_id, balance FROM accounts_data WHERE balance > 10000;",
                "SELECT * FROM dormant_flags WHERE last_activity_date < DATEADD(year, -1, GETDATE());"
            ]

            # Create a horizontal layout for sample queries
            cols = st.columns(3)
            for i, query in enumerate(sample_queries):
                with cols[i % 3]:
                    if st.button(f"Example {i + 1}", key=f"sample_sql_{i}"):
                        st.session_state.sql_query = query

    # Tab 2: Database History (using the existing implementation)
    with history_tabs[1]:
        if st.checkbox("Show Recent SQL Queries from Database", key="show_sql_history_checkbox_sqlbot"):
            try:
                history_df = get_recent_sql_history(limit=10)

                if history_df is not None and not history_df.empty:
                    for _, row in history_df.iterrows():
                        ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                        with st.expander(f"Query at {ts}: \"{row['natural_language_query'][:70]}...\""):
                            st.text_area("Natural Language:", row['natural_language_query'], height=70, disabled=True)
                            st.code(row['sql_query'], language='sql')

                            # Add re-run button for database history items
                            if st.button("🔄 Re-run", key=f"rerun_db_{_}"):
                                st.session_state.sql_query = row['sql_query']
                        st.markdown("---")  # Separator
                else:
                    st.info("No queries in database history yet.")
            except Exception as e:
                st.error(f"Error retrieving database history: {e}")


def add_query_to_history(query, results=None, explanation=None, error=None, ai_generated=False, execution_time=0.0,
                         nl_query=None):
    """
    Add a query to the session history with all relevant information.

    Args:
        query (str): The SQL query text
        results (pd.DataFrame, optional): Results of the query execution
        explanation (str, optional): Explanation of the query
        error (str, optional): Error message if query execution failed
        ai_generated (bool, optional): Whether the query was generated by AI
        execution_time (float, optional): Query execution time in seconds
        nl_query (str, optional): Natural language query that generated this SQL
    """
    # Initialize query history if it doesn't exist
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Record current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create the query history item
    query_item = {
        'query': query,
        'results': results,
        'explanation': explanation,
        'error': error,
        'timestamp': timestamp,
        'execution_time': execution_time,
        'ai_generated': ai_generated
    }

    # Add natural language query if provided
    if nl_query:
        query_item['nl_query'] = nl_query

    # Add row count if results is a DataFrame
    if results is not None and hasattr(results, 'shape'):
        query_item['row_count'] = results.shape[0]

    # Add to history (at the beginning for reverse chronological order)
    st.session_state.query_history.insert(0, query_item)

    # Limit history size to prevent memory issues
    max_history = 50  # Adjust as needed
    if len(st.session_state.query_history) > max_history:
        st.session_state.query_history = st.session_state.query_history[:max_history]

    return query_item