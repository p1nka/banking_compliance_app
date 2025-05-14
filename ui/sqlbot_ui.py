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

    # Check if there's an active database connection
    if "db_connection" in st.session_state and st.session_state["db_connection"]:
        conn = st.session_state["db_connection"]
        is_direct_connection = True
        selected_table = st.session_state.get("sql_table_schema", "Unknown table")
        st.info(f"🔗 Connected directly to database. Loaded table: **{selected_table}**")
    else:
        # Fall back to default connection
        conn = get_db_connection()
        is_direct_connection = False
        st.info(
            f"🤖 This bot queries the **default database**: `{DB_NAME}` on server `{DB_SERVER}` as configured via secrets/environment variables.")

    # Check if connection was successful
    if conn is None:
        st.warning("Cannot use SQL Bot: Database connection failed.")
        return

    # Check if LLM is available
    if not llm:
        st.warning(
            "AI Assistant (Groq/Langchain) is not available. SQL Bot will run in basic mode with limited functionality.")
        # Provide a simple SQL editor as fallback
        st.subheader("Manual SQL Query")

        # If directly connected, suggest the loaded table
        default_query = f"SELECT TOP 10 * FROM {st.session_state.get('sql_table_schema', 'accounts_data')}" if is_direct_connection else "SELECT TOP 10 * FROM accounts_data"

        manual_sql = st.text_area(
            "Enter SQL query:",
            value=default_query,
            height=150
        )

        if st.button("Execute Query", key="execute_manual_sql"):
            # Clean the SQL query to prevent syntax errors
            clean_sql = clean_sql_query(manual_sql)

            try:
                with st.spinner("Executing query..."):
                    results_df = pd.read_sql(clean_sql, conn)

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

    # Get database schema
    if is_direct_connection:
        # If using direct connection, create schema info from the current table
        try:
            # Get table structure
            cursor = conn.cursor()
            # Use SQL Server's method to get column info
            cursor.execute(
                f"SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{selected_table}'")
            columns = cursor.fetchall()

            if not columns:
                st.warning(f"Could not retrieve schema information for table '{selected_table}'")
                schema = None
            else:
                # Create schema dictionary with just the loaded table
                schema = {
                    selected_table: [(col[0], col[1]) for col in columns]
                }

            cursor.close()
        except Exception as e:
            st.warning(f"Could not retrieve schema: {e}")
            schema = None
    else:
        # Using default connection, get full schema
        schema = get_db_schema()

    if schema:
        # Format schema as text for display and prompts
        schema_text = "Database Schema for SQL Bot:\n"
        for table, columns_list in schema.items():
            schema_text += f"Table: {table}\nColumns:\n{chr(10).join([f'- {name} ({dtype})' for name, dtype in columns_list])}\n\n"

        with st.expander("Show Database Schema"):
            st.code(schema_text, language='text')

        # SQL Bot UI components
        nl_query_sqlbot = st.text_area(
            "Ask a database question:",
            placeholder=f"e.g., How many accounts in '{selected_table}' have been inactive for more than 3 years?" if is_direct_connection
            else "e.g., How many dormant accounts in 'Dubai' branch from the 'accounts_data' table?",
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
                    # If direct connection, make sure the model specifically uses the loaded table
                    if is_direct_connection:
                        nl_to_sql_prompt = PromptTemplate.from_template(
                            SQL_GENERATION_PROMPT + f"\n\nIMPORTANT: The user is currently viewing the '{selected_table}' table specifically, so ensure your SQL query focuses on this table."
                        )
                    else:
                        nl_to_sql_prompt = PromptTemplate.from_template(SQL_GENERATION_PROMPT)

                    nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

                    sql_query_raw = nl_to_sql_chain.invoke({
                        "schema": schema_text,
                        "question": nl_query_sqlbot.strip()
                    })

                    # Clean up the generated SQL with improved extraction
                    sql_query_generated = clean_sql_query(sql_query_raw)

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

                        # Add visualization options for certain types of results
                        if len(results_df.columns) >= 2 and len(results_df) > 0 and len(results_df) <= 50:
                            try:
                                import plotly.express as px

                                # Check for numeric columns that might be good for visualization
                                numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
                                all_cols = results_df.columns.tolist()

                                if numeric_cols and len(all_cols) >= 2:
                                    st.subheader("Visualize Results")

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        chart_type = st.selectbox(
                                            "Chart Type",
                                            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Box Plot"],
                                            key="vis_chart_type"
                                        )

                                    with col2:
                                        # For some charts, we need a categorical column for x-axis
                                        if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                                            x_col = st.selectbox("X-Axis", all_cols, key="vis_x_col")
                                            y_col = st.selectbox("Y-Axis", numeric_cols, key="vis_y_col")

                                            if st.button("Generate Chart", key="gen_chart_btn"):
                                                try:
                                                    if chart_type == "Bar Chart":
                                                        fig = px.bar(results_df, x=x_col, y=y_col,
                                                                     title=f"{y_col} by {x_col}")
                                                    elif chart_type == "Line Chart":
                                                        fig = px.line(results_df, x=x_col, y=y_col,
                                                                      title=f"{y_col} Trend by {x_col}")
                                                    elif chart_type == "Scatter Plot":
                                                        fig = px.scatter(results_df, x=x_col, y=y_col,
                                                                         title=f"{y_col} vs {x_col}")

                                                    st.plotly_chart(fig, use_container_width=True)
                                                except Exception as viz_e:
                                                    st.error(f"Visualization error: {viz_e}")

                                        # For pie charts
                                        elif chart_type == "Pie Chart":
                                            values_col = st.selectbox("Values", numeric_cols, key="vis_values_col")
                                            names_col = st.selectbox("Names", all_cols, key="vis_names_col")

                                            if st.button("Generate Chart", key="gen_pie_btn"):
                                                try:
                                                    fig = px.pie(results_df, values=values_col, names=names_col,
                                                                 title=f"{values_col} Distribution")
                                                    st.plotly_chart(fig, use_container_width=True)
                                                except Exception as viz_e:
                                                    st.error(f"Visualization error: {viz_e}")

                                        # For box plots
                                        elif chart_type == "Box Plot":
                                            y_col = st.selectbox("Value Column", numeric_cols, key="vis_box_y_col")

                                            # Group by is optional
                                            use_grouping = st.checkbox("Group by category", key="vis_use_grouping")
                                            if use_grouping:
                                                x_col = st.selectbox("Group by", all_cols, key="vis_box_x_col")
                                                if st.button("Generate Chart", key="gen_box_btn"):
                                                    try:
                                                        fig = px.box(results_df, y=y_col, x=x_col,
                                                                     title=f"{y_col} Distribution by {x_col}")
                                                        st.plotly_chart(fig, use_container_width=True)
                                                    except Exception as viz_e:
                                                        st.error(f"Visualization error: {viz_e}")
                                            else:
                                                if st.button("Generate Chart", key="gen_box_simple_btn"):
                                                    try:
                                                        fig = px.box(results_df, y=y_col, title=f"{y_col} Distribution")
                                                        st.plotly_chart(fig, use_container_width=True)
                                                    except Exception as viz_e:
                                                        st.error(f"Visualization error: {viz_e}")
                            except Exception as pkg_e:
                                st.info(
                                    "Visualization options not available. Please install plotly for visualization features.")

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


def clean_sql_query(raw_sql):
    """
    Clean and extract SQL query from raw text.
    Handles various formats including markdown code blocks and backticks.
    Properly escapes quotes for SQL Server compatibility.

    Args:
        raw_sql (str): The raw SQL query text to clean

    Returns:
        str: A cleaned SQL query ready for execution
    """
    if not raw_sql:
        return ""

    # First, remove any markdown code blocks
    clean_sql = re.sub(r"```sql\s*|\s*```", "", raw_sql, flags=re.IGNORECASE)

    # Handle any other code fence markers that might be present
    clean_sql = re.sub(r"```.*?\s*|\s*```", "", clean_sql, flags=re.IGNORECASE)

    # Try to extract a valid SQL statement (SELECT, INSERT, UPDATE, etc.)
    sql_pattern = re.compile(r"(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|EXEC|EXECUTE).*?(?:;|$)",
                             re.IGNORECASE | re.DOTALL)

    match = sql_pattern.search(clean_sql)
    if match:
        clean_sql = match.group(0).strip()

    # Remove trailing semicolons as they can cause issues with some SQL Server drivers
    clean_sql = clean_sql.rstrip(';')

    # Handle potential quote issues for SQL Server
    # This is a potential fix for the 'VIP' issue where nested quotes cause problems
    clean_sql = clean_sql.replace("''", "'")  # Replace double single quotes with single quotes

    return clean_sql.strip()


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