import streamlit as st
import pandas as pd
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # Added import for OpenAI chat model

from config import DB_NAME, DB_SERVER
from database.connection import get_db_connection
from database.schema import get_db_schema
from database.operations import save_sql_query_to_history, get_recent_sql_history
from ai.llm import (
    SQL_GENERATION_PROMPT,
    SQL_EXPLANATION_PROMPT,
    ADVANCED_SQL_GENERATION_PROMPT,
    ADVANCED_SQL_EXPLANATION_PROMPT,
    get_fallback_response
)


def initialize_llm():
    """
    Initialize the LLM model with proper configuration.
    Returns an initialized LLM model or None if initialization fails.
    """
    try:
        # Try to initialize the OpenAI LLM
        llm = ChatOpenAI(
            model_name="gpt-4-turbo",  # Use GPT-4 Turbo for better SQL generation
            temperature=0.1,  # Low temperature for more deterministic responses
            max_tokens=2000  # Increase token limit for detailed explanations
        )
        st.session_state["llm_available"] = True
        st.session_state["llm_instance"] = llm
        return llm
    except Exception as e:
        st.error(f"LLM initialization error: {e}")
        st.session_state["llm_available"] = False
        return None


def render_sqlbot(llm=None):
    """
    Render the SQL Bot UI for natural language to SQL queries.

    Args:
        llm: The LLM model for generating SQL
    """
    st.header("SQL Database Query Bot")

    # Initialize LLM if not provided
    if not llm:
        llm = initialize_llm()
        if llm:
            st.success("✅ AI Assistant initialized successfully")
    else:
        st.session_state["llm_available"] = True
        st.session_state["llm_instance"] = llm

    # Add mode selection for query complexity
    query_complexity = st.radio(
        "SQL Generation Mode:",
        ["Standard", "Advanced (with JOINs, Window Functions, CTEs)"],
        horizontal=True,
        index=1  # Default to Advanced
    )

    is_advanced_mode = query_complexity == "Advanced (with JOINs, Window Functions, CTEs)"

    # Add mode selection for execution
    execution_mode = st.radio(
        "Execution Mode:",
        ["Generate & Execute", "Generate Only"],
        horizontal=True,
        key="sql_execution_mode"
    )

    is_generate_only = execution_mode == "Generate Only"

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

    # Check if LLM is available - if not, try to initialize it again
    if not llm and "llm_available" not in st.session_state or not st.session_state.get("llm_available", False):
        llm = initialize_llm()
        if not llm:
            st.warning(
                "AI Assistant initialization failed. SQL Bot will run in basic mode with limited functionality.")
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

        # Store schema in session state for later use
        st.session_state['db_schema'] = schema_text

        with st.expander("Show Database Schema"):
            st.code(schema_text, language='text')

        # Add a section highlighting available advanced SQL features
        if is_advanced_mode:
            with st.expander("Advanced SQL Features Available"):
                st.markdown("""
                **This bot can generate SQL with the following advanced features:**

                * **JOIN Operations**: INNER, LEFT, RIGHT, FULL JOINs for relating multiple tables
                * **Window Functions**: ROW_NUMBER(), RANK(), DENSE_RANK(), LEAD(), LAG() for analytical operations
                * **Common Table Expressions (CTEs)**: WITH clauses for complex multi-step logic
                * **Subqueries and Derived Tables**: For complex nested operations
                * **PIVOT Operations**: For transforming row data into columns
                * **Advanced Aggregation**: GROUP BY with aggregate functions and HAVING clauses
                * **Conditional Logic**: CASE statements for complex branching
                * **Pagination**: TOP/OFFSET-FETCH for limiting results
                * **Temporal Functions**: For date/time analysis

                Simply describe what you need in natural language, and the bot will generate the appropriate advanced SQL query!
                """)

        # SQL Bot UI components
        example_placeholder = (
            f"e.g., Rank the {selected_table} by inactivity duration and show the top 5 most inactive accounts"
            if is_direct_connection and is_advanced_mode
            else "e.g., Find accounts in 'accounts_data' that have been inactive for more than 3 years, and rank them by balance"
        )

        nl_query_sqlbot = st.text_area(
            "Ask a database question:",
            placeholder=example_placeholder,
            height=100,
            key="sql_bot_nl_query_input"
        )

        # Single button that adapts based on selected execution mode
        button_text = "Generate SQL" if is_generate_only else "Generate & Execute SQL"
        generate_button = st.button(button_text, key="sql_bot_action_button")

        # Initialize variables
        sql_query_generated = None

        if generate_button and nl_query_sqlbot:
            # Generate SQL
            try:
                with st.spinner(f"🤖 Converting natural language to {'advanced ' if is_advanced_mode else ''}SQL..."):
                    # Choose the appropriate prompt based on mode
                    prompt_template = ADVANCED_SQL_GENERATION_PROMPT if is_advanced_mode else SQL_GENERATION_PROMPT

                    # If direct connection, make sure the model specifically uses the loaded table
                    if is_direct_connection:
                        nl_to_sql_prompt = PromptTemplate.from_template(
                            prompt_template + f"\n\nIMPORTANT: The user is currently viewing the '{selected_table}' table specifically, so ensure your SQL query focuses on this table."
                        )
                    else:
                        nl_to_sql_prompt = PromptTemplate.from_template(prompt_template)

                    # Updated to use modern langchain methods
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

            # Store in session state for later use
            if sql_query_generated:
                st.session_state.last_nl_query_sqlbot = nl_query_sqlbot
                st.session_state.generated_sql_sqlbot = sql_query_generated

                # Display the generated SQL
                st.subheader("Generated SQL Query")
                st.code(sql_query_generated, language='sql')

                # Save query to history
                try:
                    if save_sql_query_to_history(nl_query_sqlbot, sql_query_generated):
                        st.success("Query saved to history.")
                    else:
                        st.warning("Failed to save query to history.")
                except Exception as e:
                    st.warning(f"Error saving to history: {e}")

                # Add to local history for the current session
                add_query_to_history(
                    query=sql_query_generated,
                    explanation=f"Generated from: '{nl_query_sqlbot}'",
                    ai_generated=True
                )

                # Execute SQL if in execution mode
                if not is_generate_only:
                    try:
                        with st.spinner("⏳ Executing query..."):
                            # Use pandas to read and display the results
                            results_df = pd.read_sql(sql_query_generated, conn)

                        # Display results
                        st.subheader("Query Results")
                        if not results_df.empty:
                            st.dataframe(results_df)
                            st.info(f"Query returned {len(results_df)} rows.")

                            # Update the history entry with results
                            if st.session_state.query_history:
                                st.session_state.query_history[0]['results'] = results_df
                                st.session_state.query_history[0]['row_count'] = len(results_df)

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
                                                            fig = px.box(results_df, y=y_col,
                                                                         title=f"{y_col} Distribution")
                                                            st.plotly_chart(fig, use_container_width=True)
                                                        except Exception as viz_e:
                                                            st.error(f"Visualization error: {viz_e}")
                                except Exception as pkg_e:
                                    st.info(
                                        "Visualization options not available. Please install plotly for visualization features.")

                                # Add enhanced insights button
                                if st.button("📊 Generate Data Insights", key="gen_insights_btn"):
                                    try:
                                        with st.spinner("🔍 Analyzing data patterns..."):
                                            # Create a data insights prompt
                                            data_insights_prompt = f"""
                                            You are a data analyst examining SQL query results. Analyze the following data summary and provide 3-5 key insights, in bullet point format:

                                            Table summary: {results_df.describe().to_string()}

                                            Column names: {list(results_df.columns)}
                                            Sample data (first 5 rows): {results_df.head(5).to_dict()}

                                            Original query: "{sql_query_generated}"
                                            Natural language request: "{nl_query_sqlbot}"

                                            Provide clear, actionable insights focusing on patterns, anomalies, or business implications.
                                            """

                                            # Call the LLM for insights
                                            insights_prompt = PromptTemplate.from_template(data_insights_prompt)
                                            insights_chain = insights_prompt | llm | StrOutputParser()

                                            insights = insights_chain.invoke({})

                                            # Display insights
                                            st.subheader("Data Insights")
                                            st.markdown(insights)
                                    except Exception as insights_e:
                                        st.error(f"Insights generation error: {insights_e}")

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
                        # Update history with error
                        if st.session_state.query_history:
                            st.session_state.query_history[0]['error'] = str(e)
                else:
                    # Add an option to execute even in Generate Only mode
                    if st.button("Execute Generated Query", key="execute_generated_query"):
                        try:
                            with st.spinner("⏳ Executing query..."):
                                results_df = pd.read_sql(sql_query_generated, conn)

                            # Display results
                            st.subheader("Query Results")
                            if not results_df.empty:
                                st.dataframe(results_df)
                                st.info(f"Query returned {len(results_df)} rows.")

                                # Update the history entry with results
                                if st.session_state.query_history:
                                    st.session_state.query_history[0]['results'] = results_df
                                    st.session_state.query_history[0]['row_count'] = len(results_df)

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
                            # Update history with error
                            if st.session_state.query_history:
                                st.session_state.query_history[0]['error'] = str(e)

        # Initialize session state variables if not already set
        if 'last_nl_query_sqlbot' not in st.session_state:
            st.session_state.last_nl_query_sqlbot = ""
        if 'generated_sql_sqlbot' not in st.session_state:
            st.session_state.generated_sql_sqlbot = None

        # Enhanced SQL Explanation button
        if st.session_state.get('generated_sql_sqlbot'):
            if st.button("Explain Generated Query", key="analyze_sql_button_sqlbot"):
                try:
                    with st.spinner("🧠 Analyzing advanced SQL query..."):
                        # Use the advanced explanation prompt for more detailed analysis
                        explanation_prompt = PromptTemplate.from_template(
                            ADVANCED_SQL_EXPLANATION_PROMPT if is_advanced_mode else SQL_EXPLANATION_PROMPT
                        )
                        sql_explanation_chain = explanation_prompt | llm | StrOutputParser()

                        explanation = sql_explanation_chain.invoke({
                            "sql_query": st.session_state.generated_sql_sqlbot,
                            "schema": schema_text
                        })
                        st.subheader("Query Analysis")
                        st.markdown(explanation)

                        # Update the history entry with explanation
                        if st.session_state.query_history:
                            st.session_state.query_history[0]['explanation'] = explanation
                except Exception as e:
                    st.error(f"Query explanation error: {e}")
                    fallback_explanation = get_fallback_response("sql_explanation")
                    st.warning("Using fallback explanation due to AI error.")
                    st.markdown(fallback_explanation)

        # Add option to optimize the query if it has been generated
        if st.session_state.get('generated_sql_sqlbot'):
            if st.button("🚀 Optimize Generated Query", key="optimize_sql_button"):
                try:
                    with st.spinner("⚙️ Optimizing SQL query for performance..."):
                        # Create optimization prompt
                        optimization_prompt = """
                        You are a database performance expert. Review and optimize the following SQL query for better performance.
                        The optimization should maintain the exact same result set but improve execution speed.

                        Original SQL Query:
                        ```sql
                        {sql_query}
                        ```

                        Database Schema:
                        {schema}

                        Please provide:
                        1. The optimized SQL query
                        2. A brief explanation of the optimizations made
                        3. Why these optimizations improve performance

                        Optimized Query:
                        """

                        optimize_prompt = PromptTemplate.from_template(optimization_prompt)
                        sql_optimization_chain = optimize_prompt | llm | StrOutputParser()

                        optimization_result = sql_optimization_chain.invoke({
                            "sql_query": st.session_state.generated_sql_sqlbot,
                            "schema": schema_text
                        })

                        # Extract just the SQL from the result
                        optimized_sql_match = re.search(r"```sql\s*(.*?)\s*```", optimization_result, re.DOTALL)
                        if optimized_sql_match:
                            optimized_sql = optimized_sql_match.group(1).strip()
                        else:
                            # Try a different pattern if the first one fails
                            optimized_sql_match = re.search(r"Optimized Query:\s*(.*?)(?:\n\n|$)", optimization_result,
                                                            re.DOTALL)
                            if optimized_sql_match:
                                optimized_sql = optimized_sql_match.group(1).strip()
                            else:
                                optimized_sql = None

                        # Display optimization results
                        st.subheader("Query Optimization")

                        # Display full optimization details
                        st.markdown(optimization_result)

                        # If valid SQL was extracted, allow execution of optimized query
                        if optimized_sql and optimized_sql.lower().strip().startswith("select"):
                            st.session_state.optimized_sql = optimized_sql
                            st.subheader("Optimized SQL Query")
                            st.code(optimized_sql, language='sql')

                            if st.button("Execute Optimized Query", key="execute_optimized_query"):
                                try:
                                    with st.spinner("⏳ Executing optimized query..."):
                                        opt_results_df = pd.read_sql(optimized_sql, conn)

                                    st.subheader("Optimized Query Results")
                                    if not opt_results_df.empty:
                                        st.dataframe(opt_results_df)
                                        st.info(f"Optimized query returned {len(opt_results_df)} rows.")

                                        # Add to history
                                        add_query_to_history(
                                            query=optimized_sql,
                                            results=opt_results_df,
                                            explanation="Optimized version of previous query",
                                            ai_generated=True
                                        )

                                        # CSV download button for results
                                        csv_data = opt_results_df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label="Download Optimized Results as CSV",
                                            data=csv_data,
                                            file_name=f"optimized_sql_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                            mime="text/csv"
                                        )
                                    else:
                                        st.info("Optimized query executed successfully but returned no results.")
                                except Exception as e:
                                    st.error(f"Optimized query execution error: {e}")
                except Exception as e:
                    st.error(f"Query optimization error: {e}")

        # Show query history
        show_query_history()
    else:
        st.warning("Could not retrieve database schema. SQL Bot is limited.")


def clean_sql_query(sql_text):
    """
    Clean and extract valid SQL from raw text.

    Args:
        sql_text: Raw text that may contain SQL

    Returns:
        str: Cleaned SQL query
    """
    if not sql_text:
        return None

    # Remove markdown code blocks if present
    sql_text = re.sub(r"^```sql\s*|\s*```$", "", sql_text, flags=re.MULTILINE).strip()

    # Look for SELECT statement
    match = re.search(r"SELECT.*", sql_text, re.IGNORECASE | re.DOTALL)
    if match:
        sql_query = match.group(0).strip()
        return sql_query

    return sql_text


def generate_sql_only(nl_query, schema_text, llm, is_advanced_mode=True, selected_table=None):
    """
    Generate SQL from natural language without executing it.

    Args:
        nl_query: Natural language query from user
        schema_text: Database schema information
        llm: Language model for SQL generation
        is_advanced_mode: Whether to use advanced SQL features
        selected_table: Optional specific table to focus on

    Returns:
        Generated SQL query or None if generation fails
    """
    try:
        # Choose the appropriate prompt based on mode
        prompt_template = ADVANCED_SQL_GENERATION_PROMPT if is_advanced_mode else SQL_GENERATION_PROMPT

        # If table specified, make sure the model focuses on it
        if selected_table:
            nl_to_sql_prompt = PromptTemplate.from_template(
                prompt_template + f"\n\nIMPORTANT: The user is currently viewing the '{selected_table}' table specifically, so ensure your SQL query focuses on this table."
            )
        else:
            nl_to_sql_prompt = PromptTemplate.from_template(prompt_template)

        nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

        sql_query_raw = nl_to_sql_chain.invoke({
            "schema": schema_text,
            "question": nl_query.strip()
        })

        # Clean up the generated SQL
        sql_query_generated = clean_sql_query(sql_query_raw)

        # Validate the generated query is a SELECT statement
        if not sql_query_generated or not sql_query_generated.lower().strip().startswith("select"):
            print("Generated text does not start with SELECT or is empty.")
            return None

        return sql_query_generated

    except Exception as e:
        print(f"SQL generation error: {e}")
        return None


def show_query_history():
    """
    Display the history of SQL queries executed in the application with AI enhancement.
    Provides rich insights and explanations when AI is available, with graceful fallback.
    """
    import streamlit as st
    import pandas as pd
    from datetime import datetime

    # Initialize query history in session state if it doesn't exist
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Check if there's any history to display
    if st.session_state.query_history:
        st.subheader("Query History")

        # Add filtering options
        col1, col2 = st.columns([1, 2])
        with col1:
            filter_type = st.selectbox(
                "Filter by type:",
                ["All Queries", "AI Generated", "Manual", "Successful", "Failed"]
            )

        with col2:
            if len(st.session_state.query_history) > 0:
                search_term = st.text_input("Search in queries:", "")

        # Filter the history based on selection
        filtered_history = st.session_state.query_history

        if filter_type == "AI Generated":
            filtered_history = [q for q in filtered_history if q.get('ai_generated', False)]
        elif filter_type == "Manual":
            filtered_history = [q for q in filtered_history if not q.get('ai_generated', False)]
        elif filter_type == "Successful":
            filtered_history = [q for q in filtered_history if not q.get('error')]
        elif filter_type == "Failed":
            filtered_history = [q for q in filtered_history if q.get('error')]

        # Apply search filter if provided
        if 'search_term' in locals() and search_term:
            filtered_history = [q for q in filtered_history if search_term.lower() in q.get('query', '').lower()]

        # Display metrics
        if len(filtered_history) > 0:
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

            with metrics_col1:
                st.metric("Total Queries", len(filtered_history))

            with metrics_col2:
                ai_generated = sum(1 for q in filtered_history if q.get('ai_generated', False))
                st.metric("AI Generated", ai_generated)

            with metrics_col3:
                success_rate = sum(1 for q in filtered_history if not q.get('error')) / len(filtered_history) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")

            with metrics_col4:
                if any('execution_time' in q for q in filtered_history):
                    avg_time = sum(
                        float(q.get('execution_time', 0)) for q in filtered_history if 'execution_time' in q) / len(
                        filtered_history)
                    st.metric("Avg. Run Time", f"{avg_time:.2f}s")

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
                tabs = st.tabs(["Query", "Results", "Explanation", "Metadata"])

                with tabs[0]:  # Query tab
                    st.code(query_item.get('query', 'No query available'), language='sql')

                    # Add copy button functionality
                    if st.button("📋 Copy SQL", key=f"copy_btn_{i}"):
                        st.toast("SQL copied to clipboard!")
                        st.session_state.clipboard = query_item.get('query', '')

                    # Add re-run button
                    if st.button("🔄 Re-run Query", key=f"rerun_btn_{i}"):
                        st.session_state.rerun_query = query_item.get('query', '')
                        st.experimental_rerun()

                with tabs[1]:  # Results tab
                    if 'error' in query_item and query_item['error']:
                        st.error(f"Error: {query_item['error']}")
                    elif 'results' in query_item and query_item['results'] is not None:
                        # Check if results is a DataFrame
                        if isinstance(query_item['results'], pd.DataFrame):
                            if not query_item['results'].empty:
                                st.dataframe(query_item['results'], use_container_width=True)

                                # Add download button for CSV
                                csv = query_item['results'].to_csv(index=False)
                                st.download_button(
                                    label="📥 Download as CSV",
                                    data=csv,
                                    file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key=f"download_btn_{i}"
                                )
                            else:
                                st.info("Query returned no results.")
                        else:
                            st.write(query_item['results'])
                    else:
                        st.info("No results available.")

                with tabs[2]:  # Explanation tab
                    if 'explanation' in query_item and query_item['explanation']:
                        st.write(query_item['explanation'])

                        # Add AI enhancement option if LLM is available
                        if st.session_state.get('llm_available', False) and st.session_state.get('llm_instance'):
                            if st.button("🧠 Generate Enhanced Explanation", key=f"enhance_btn_{i}"):
                                try:
                                    with st.spinner("Generating enhanced explanation..."):
                                        # Get the schema from session state
                                        schema = st.session_state.get('db_schema', "Schema not available")

                                        # Create the explanation prompt
                                        explanation_prompt = PromptTemplate.from_template(
                                            ADVANCED_SQL_EXPLANATION_PROMPT
                                        )

                                        # Get the LLM from session state
                                        llm = st.session_state['llm_instance']

                                        # Create the chain
                                        explanation_chain = explanation_prompt | llm | StrOutputParser()

                                        # Generate enhanced explanation
                                        explanation = explanation_chain.invoke({
                                            "sql_query": query_item.get('query', ''),
                                            "schema": schema
                                        })

                                        # Update the query item with new explanation
                                        query_item['explanation'] = explanation
                                        st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Explanation error: {e}")
                    else:
                        st.info("No explanation available.")

                        # Add AI explanation generation if LLM is available
                        if st.session_state.get('llm_available', False) and st.session_state.get('llm_instance'):
                            if st.button("🧠 Generate Explanation", key=f"gen_explain_btn_{i}"):
                                try:
                                    with st.spinner("Generating explanation..."):
                                        # Get the schema from session state
                                        schema = st.session_state.get('db_schema', "Schema not available")

                                        # Create the explanation prompt
                                        explanation_prompt = PromptTemplate.from_template(
                                            SQL_EXPLANATION_PROMPT
                                        )

                                        # Get the LLM from session state
                                        llm = st.session_state['llm_instance']

                                        # Create the chain
                                        explanation_chain = explanation_prompt | llm | StrOutputParser()

                                        # Generate explanation
                                        explanation = explanation_chain.invoke({
                                            "sql_query": query_item.get('query', ''),
                                            "schema": schema
                                        })

                                        # Update the query item with new explanation
                                        query_item['explanation'] = explanation
                                        st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Explanation error: {e}")

                with tabs[3]:  # Metadata tab
                    metadata_col1, metadata_col2 = st.columns(2)

                    with metadata_col1:
                        st.write("**Query Details:**")
                        st.text(f"Timestamp: {query_item.get('timestamp', 'N/A')}")
                        st.text(f"Execution time: {query_item.get('execution_time', 'N/A')} seconds")
                        st.text(f"Generated by: {'AI Assistant' if query_item.get('ai_generated', False) else 'User'}")

                    with metadata_col2:
                        st.write("**Performance:**")
                        if 'row_count' in query_item:
                            st.text(f"Rows returned: {query_item.get('row_count', 0)}")
                        if 'performance_notes' in query_item:
                            st.text(f"Notes: {query_item.get('performance_notes', '')}")

                    # Add insights section if AI is available
                    if st.session_state.get('llm_available', False) and st.session_state.get('llm_instance'):
                        if st.button("🔍 Generate Query Insights", key=f"insights_btn_{i}"):
                            try:
                                with st.spinner("Analyzing query..."):
                                    # Get the LLM instance
                                    llm = st.session_state['llm_instance']

                                    # Custom prompt for insights
                                    insights_prompt_template = """
                                    You are a database performance expert. Analyze this SQL query and provide brief insights about:
                                    1. Query structure and best practices
                                    2. Potential performance optimizations
                                    3. Security considerations
                                    Keep your analysis concise and actionable.

                                    SQL Query:
                                    ```sql
                                    {query}
                                    ```

                                    Insights:
                                    """

                                    # Create the prompt
                                    insights_prompt = PromptTemplate.from_template(insights_prompt_template)

                                    # Create the chain
                                    insights_chain = insights_prompt | llm | StrOutputParser()

                                    # Generate insights
                                    insights = insights_chain.invoke({
                                        "query": query_item.get('query', '')
                                    })

                                    # Display insights
                                    st.subheader("Query Insights")
                                    st.write(insights)

                                    # Store insights for future reference
                                    query_item['insights'] = insights
                            except Exception as e:
                                st.error(f"Insights generation error: {e}")
    else:
        # Display a message if no queries have been run
        st.info("No SQL queries have been executed yet.")

        # Provide a helpful tip based on whether AI is available
        if st.session_state.get('llm_available', False):
            st.success("Tip: You can ask questions in natural language, and the AI will generate SQL for you!")

            # Add sample questions to get started
            st.write("**Sample questions you can ask:**")
            sample_questions = [
                "Show me all accounts with balances over $10,000",
                "Find dormant accounts that have had no activity in the last year",
                "List the most recent transactions for high-risk customers",
                "Show me accounts with suspicious activity patterns"
            ]

            for i, question in enumerate(sample_questions):
                if st.button(question, key=f"sample_q_{i}"):
                    st.session_state.user_question = question
                    st.experimental_rerun()
        else:
            st.warning("Note: AI assistance is currently disabled. You can still enter SQL queries manually.")

            # Add sample SQL queries to get started
            st.write("**Sample SQL queries you can try:**")
            sample_queries = [
                "SELECT * FROM accounts_data LIMIT 10;",
                "SELECT account_id, balance FROM accounts_data WHERE balance > 10000;",
                "SELECT * FROM dormant_flags WHERE last_activity_date < DATEADD(year, -1, GETDATE());"
            ]

            for i, query in enumerate(sample_queries):
                if st.button(f"Example {i + 1}", key=f"sample_sql_{i}"):
                    st.session_state.sql_query = query
                    st.experimental_rerun()


# Helper function to add a query to the history
def add_query_to_history(query, results=None, explanation=None, error=None, ai_generated=False):
    """
    Add a query to the history with all relevant information.

    Args:
        query (str): The SQL query text
        results (pd.DataFrame, optional): Results of the query execution
        explanation (str, optional): AI-generated or manual explanation of the query
        error (str, optional): Error message if query execution failed
        ai_generated (bool, optional): Whether the query was generated by AI
    """
    import streamlit as st
    from datetime import datetime
    import time

    # Initialize query history if it doesn't exist
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    # Record execution metrics
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start execution time tracking
    start_time = time.time()

    # Create the query history item
    query_item = {
        'query': query,
        'results': results,
        'explanation': explanation,
        'error': error,
        'timestamp': timestamp,
        'ai_generated': ai_generated,
        'execution_time': round(time.time() - start_time, 3)
    }

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


def execute_sql_query(query, connection):
    """
    Execute a SQL query and store it in the history.

    Args:
        query: SQL query string to execute
        connection: Database connection object

    Returns:
        DataFrame with query results or None if execution failed
    """
    import streamlit as st
    import pandas as pd
    import time
    from datetime import datetime

    if 'query_history' not in st.session_state:
        st.session_state.query_history = []

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        results = pd.read_sql(query, connection)
        execution_time = round(time.time() - start_time, 3)

        # Store in history
        history_entry = {
            'query': query,
            'results': results,
            'timestamp': timestamp,
            'execution_time': execution_time,
            'error': None,
            'row_count': len(results),
            'ai_generated': False
        }

        st.session_state.query_history.insert(0, history_entry)

        return results

    except Exception as e:
        execution_time = round(time.time() - start_time, 3)

        # Store error in history
        history_entry = {
            'query': query,
            'results': None,
            'timestamp': timestamp,
            'execution_time': execution_time,
            'error': str(e),
            'ai_generated': False
        }

        st.session_state.query_history.insert(0, history_entry)

        raise e


# Add a function for automated performance analysis
def analyze_query_performance(query, schema_text, llm):
    """
    Analyze SQL query for performance issues and suggest optimizations.

    Args:
        query: SQL query string to analyze
        schema_text: Database schema as text
        llm: LLM instance

    Returns:
        String with performance analysis
    """
    try:
        # Create performance analysis prompt
        performance_prompt_template = """
        You are a database performance expert. Analyze the following SQL query for potential performance issues:

        ```sql
        {sql_query}
        ```

        Database Schema:
        {schema}

        Provide a concise analysis of:
        1. Potential performance bottlenecks
        2. Missing indexes or optimization opportunities
        3. Best practices that could be applied
        4. Alternative query structures that might be more efficient

        Format your response as a structured performance report with clear recommendations.
        """

        performance_prompt = PromptTemplate.from_template(performance_prompt_template)
        performance_chain = performance_prompt | llm | StrOutputParser()

        performance_analysis = performance_chain.invoke({
            "sql_query": query,
            "schema": schema_text
        })

        return performance_analysis
    except Exception as e:
        return f"Performance analysis failed: {str(e)}"