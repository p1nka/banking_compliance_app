import streamlit as st
import pandas as pd
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import DB_NAME, DB_SERVER
from database.connection import get_db_connection
from database.schema import get_db_schema
from database.operations import save_sql_query_to_history, get_recent_sql_history
from ai.llm import SQL_GENERATION_PROMPT, SQL_EXPLANATION_PROMPT, get_fallback_response


def render_sqlbot(llm):
    """
    Render the SQL Bot UI for natural language to SQL queries.

    Args:
        llm: The LLM model for generating SQL
    """
    st.header("SQL Database Query Bot")

    # Explicitly state which database is being queried
    st.info(
        f"🤖 This bot queries the **default database**: `{DB_NAME}` on server `{DB_SERVER}` as configured via secrets/environment variables.")
    st.caption("*(This is separate from the 'Load Data' feature, which brings data into the app's memory.)*")

    # Check prerequisites
    conn = get_db_connection()
    if conn is None:
        st.warning("Cannot use SQL Bot: Default database connection failed.")
        return

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
                    results_df = pd.read_sql(manual_sql, conn)

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

        # Show query history
        show_query_history()
        return

    # Fetch schema for the default database
    schema = get_db_schema()

    if schema:
        # Format schema as text for display and prompts
        schema_text = "Database Schema for SQL Bot:\n"
        for table, columns_list in schema.items():
            schema_text += f"Table: {table}\nColumns:\n{chr(10).join([f'- {name} ({dtype})' for name, dtype in columns_list])}\n\n"

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
                        results_df = pd.read_sql(sql_query_generated, conn)

                    # Save query to history
                    try:
                        if save_sql_query_to_history(nl_query_sqlbot, sql_query_generated):
                            st.success("Query saved to history.")
                        else:
                            st.warning("Failed to save query to history.")
                    except Exception as e:
                        st.warning(f"Error saving to history: {e}")

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
                except Exception as e:
                    st.error(f"Query explanation error: {e}")
                    fallback_explanation = get_fallback_response("sql_explanation")
                    st.warning("Using fallback explanation due to AI error.")
                    st.markdown(fallback_explanation)

        # Show query history
        show_query_history()
    else:
        st.warning("Could not retrieve database schema. SQL Bot is limited.")


def show_query_history():
    """Display the SQL query history."""
    st.subheader("Query History")
    if st.checkbox("Show Recent SQL Queries from History", key="show_sql_history_checkbox_sqlbot"):
        history_df = get_recent_sql_history(limit=10)

        if history_df is not None and not history_df.empty:
            for _, row in history_df.iterrows():
                ts = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                with st.expander(f"Query at {ts}: \"{row['natural_language_query'][:70]}...\""):
                    st.text_area("Natural Language:", row['natural_language_query'], height=70, disabled=True)
                    st.code(row['sql_query'], language='sql')
                st.markdown("---")  # Separator
        else:
            st.info("No queries in history yet.")