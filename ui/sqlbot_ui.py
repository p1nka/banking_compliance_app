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
from ai.visualizations import create_insights_chart
import plotly.express as px


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

    # Show connection status
    st.info(f"🤖 Connected to database: `{DB_NAME}` on server `{DB_SERVER}`")

    # Get database schema
    schema = get_db_schema()
    if schema:
        schema_text = format_schema_for_display(schema)
        
        with st.expander("Show Database Schema"):
            st.code(schema_text, language='text')

        # SQL Bot UI components
        example_placeholder = "e.g., Show me all accounts that require Article 3 process"

        nl_query_sqlbot = st.text_area(
            "Ask a database question:",
            placeholder=example_placeholder,
            height=100,
            key="sql_bot_nl_query_input"
        )

        # Single button that adapts based on selected execution mode
        button_text = "Generate SQL" if is_generate_only else "Generate & Execute SQL"
        generate_button = st.button(button_text, key="sql_bot_action_button", type="primary")

        if generate_button and nl_query_sqlbot:
            process_sql_query(nl_query_sqlbot, schema_text, llm, conn, is_generate_only, is_advanced_mode)

        # Show generated SQL if available and add execute button for generate-only mode
        if 'generated_sql_sqlbot' in st.session_state and st.session_state.generated_sql_sqlbot:
            st.subheader("Generated SQL Query")
            st.code(st.session_state.generated_sql_sqlbot, language='sql')
            
            # Add execute button if in generate-only mode
            if is_generate_only:
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Execute Generated Query", key="execute_generated_query", type="secondary"):
                        execute_sql_query(st.session_state.generated_sql_sqlbot, conn)
                with col2:
                    if st.button("Clear Generated Query", key="clear_generated_query"):
                        if 'generated_sql_sqlbot' in st.session_state:
                            del st.session_state['generated_sql_sqlbot']
                        if 'last_nl_query_sqlbot' in st.session_state:
                            del st.session_state['last_nl_query_sqlbot']
                        st.rerun()

        # Show query history
        show_query_history()
        
        # Add quick examples
        show_example_queries()
        
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

    if st.button("Execute Query", key="execute_manual_sql", type="primary"):
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

            # Save to history (with better error handling)
            save_query_to_history(nl_query, sql_query_generated)

            # Execute SQL if in execution mode
            if not is_generate_only:
                execute_sql_query(sql_query_generated, conn)

    except Exception as e:
        st.error(f"Error processing query: {e}")


def generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode=False):
    """Generate SQL from natural language using enhanced LLM."""
    return enhanced_generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode)
    


def execute_sql_query(sql_query, conn):
    """Execute SQL query with enhanced datetime handling."""
    return enhanced_execute_sql_query(sql_query, conn)

def show_query_history():
    """Display query history."""
    st.subheader("📜 Query History")
    
    # Show both session history and database history
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Current Session**")
        if 'query_history' not in st.session_state or not st.session_state.query_history:
            st.info("No queries in current session.")
        else:
            for i, item in enumerate(st.session_state.query_history[:3]):  # Show last 3
                with st.expander(f"Query {i+1} - {item['timestamp']}", expanded=False):
                    st.code(item['query'], language='sql')
                    
                    if item['error']:
                        st.error(f"Error: {item['error']}")
                    elif item['results'] is not None and not item['results'].empty:
                        st.success(f"✅ Returned {len(item['results'])} rows")
                        if st.button(f"View Results {i+1}", key=f"view_results_{i}"):
                            st.dataframe(item['results'])
                    else:
                        st.info("No results returned")

    with col2:
        st.write("**Database History**")
        try:
            db_history = get_recent_sql_history(5)
            if db_history is not None and not db_history.empty:
                for idx, row in db_history.iterrows():
                    with st.expander(f"DB Query - {row['timestamp']}", expanded=False):
                        st.text(f"Question: {row['natural_language_query']}")
                        st.code(row['sql_query'], language='sql')
            else:
                st.info("No database history available.")
        except Exception as e:
            st.info("Database history not accessible.")

    # Button to clear history
    if st.button("🗑️ Clear Session History", key="clear_history_btn"):
        st.session_state.query_history = []
        st.rerun()


def auto_generate_visualizations(results_df):
    """
    Automatically generate appropriate visualizations for SQL query results.
    
    Args:
        results_df (pd.DataFrame): The query results to visualize
    """
    if results_df.empty or len(results_df) == 0:
        return
    
    st.subheader("📈 Auto-Generated Visualizations")
    
    # Analyze the data structure
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
    date_cols = results_df.select_dtypes(include=['datetime64[ns]', 'datetime']).columns.tolist()
    
    # Filter out ID columns and columns with too many unique values for categorical analysis
    categorical_cols = [col for col in categorical_cols 
                       if not col.lower().endswith('_id') 
                       and not col.lower().startswith('id')
                       and results_df[col].nunique() <= 20
                       and results_df[col].nunique() > 1]
    
    visualizations_created = 0
    max_visualizations = 3  # Limit to avoid overwhelming the user
    
    try:
        # 1. Create pie chart for categorical data (if suitable)
        if categorical_cols and visualizations_created < max_visualizations:
            best_cat_col = None
            best_unique_count = 0
            
            # Find the best categorical column (reasonable number of unique values)
            for col in categorical_cols:
                unique_count = results_df[col].nunique()
                if 2 <= unique_count <= 10 and unique_count > best_unique_count:
                    best_cat_col = col
                    best_unique_count = unique_count
            
            if best_cat_col:
                try:
                    fig = create_insights_chart(
                        results_df, 
                        labels=best_cat_col, 
                        chart_type='pie',
                        title=f"Distribution of {best_cat_col.replace('_', ' ').title()}"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"pie_{best_cat_col}")
                        visualizations_created += 1
                        st.caption(f"📊 Pie chart showing the distribution of values in '{best_cat_col}'")
                except Exception as e:
                    st.write(f"Could not create pie chart: {e}")

        # 2. Create bar chart for categorical vs numeric data
        if categorical_cols and numeric_cols and visualizations_created < max_visualizations:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Check if the combination makes sense
            if results_df[cat_col].nunique() <= 15:
                try:
                    # Aggregate numeric data by category
                    agg_data = results_df.groupby(cat_col)[num_col].agg(['count', 'mean', 'sum']).reset_index()
                    
                    # Choose the most appropriate aggregation
                    if 'balance' in num_col.lower() or 'amount' in num_col.lower():
                        y_col = 'sum'
                        title = f"Total {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
                    else:
                        y_col = 'mean'
                        title = f"Average {num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
                    
                    fig = px.bar(
                        agg_data, 
                        x=cat_col, 
                        y=y_col,
                        title=title,
                        color=y_col,
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_{cat_col}_{num_col}")
                    visualizations_created += 1
                    st.caption(f"📊 Bar chart showing {y_col} of '{num_col}' grouped by '{cat_col}'")
                except Exception as e:
                    st.write(f"Could not create bar chart: {e}")

        # 3. Create histogram for numeric data
        if numeric_cols and visualizations_created < max_visualizations:
            num_col = numeric_cols[0]
            
            # Skip if too few data points
            if len(results_df) > 5:
                try:
                    fig = px.histogram(
                        results_df, 
                        x=num_col,
                        title=f"Distribution of {num_col.replace('_', ' ').title()}",
                        nbins=min(30, max(10, len(results_df) // 5))
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_{num_col}")
                    visualizations_created += 1
                    st.caption(f"📊 Histogram showing the distribution of values in '{num_col}'")
                except Exception as e:
                    st.write(f"Could not create histogram: {e}")

        # 4. Create time series if date column exists
        if date_cols and len(results_df) > 1 and visualizations_created < max_visualizations:
            date_col = date_cols[0]
            
            try:
                # Sort by date
                df_sorted = results_df.sort_values(date_col)
                
                if numeric_cols:
                    # Time series with numeric value
                    num_col = numeric_cols[0]
                    fig = px.line(
                        df_sorted, 
                        x=date_col, 
                        y=num_col,
                        title=f"{num_col.replace('_', ' ').title()} Over Time"
                    )
                else:
                    # Just count occurrences over time
                    date_counts = df_sorted[date_col].value_counts().sort_index().reset_index()
                    date_counts.columns = [date_col, 'count']
                    fig = px.line(
                        date_counts, 
                        x=date_col, 
                        y='count',
                        title=f"Activity Over Time"
                    )
                
                st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{date_col}")
                visualizations_created += 1
                st.caption(f"📊 Time series chart showing data trends over '{date_col}'")
            except Exception as e:
                st.write(f"Could not create time series: {e}")

        # 5. Create correlation heatmap if multiple numeric columns
        if len(numeric_cols) >= 2 and len(results_df) > 10 and visualizations_created < max_visualizations:
            try:
                # Select up to 5 numeric columns to avoid clutter
                selected_numeric = numeric_cols[:5]
                correlation_matrix = results_df[selected_numeric].corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    title="Correlation Matrix of Numeric Columns",
                    color_continuous_scale='RdBu',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
                visualizations_created += 1
                st.caption("📊 Correlation heatmap showing relationships between numeric columns")
            except Exception as e:
                st.write(f"Could not create correlation heatmap: {e}")

        # 6. Create scatter plot if we have exactly 2 numeric columns
        if len(numeric_cols) == 2 and len(results_df) > 5 and visualizations_created < max_visualizations:
            try:
                x_col, y_col = numeric_cols[0], numeric_cols[1]
                color_col = categorical_cols[0] if categorical_cols else None
                
                fig = px.scatter(
                    results_df,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
                    hover_data=results_df.columns
                )
                st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_col}_{y_col}")
                visualizations_created += 1
                st.caption(f"📊 Scatter plot showing relationship between '{x_col}' and '{y_col}'")
            except Exception as e:
                st.write(f"Could not create scatter plot: {e}")

        # Show summary if no visualizations were created
        if visualizations_created == 0:
            st.info("💡 No suitable visualizations could be automatically generated for this data. Try:")
            st.write("- Queries that return categorical data for pie charts")
            st.write("- Queries with numeric data for histograms") 
            st.write("- Queries with dates for time series")
            st.write("- Queries combining categories and numbers for bar charts")
            
            # Show basic data summary instead
            with st.expander("📊 Data Summary"):
                st.write("**Data Types:**")
                st.write(f"- Numeric columns: {len(numeric_cols)} ({', '.join(numeric_cols) if numeric_cols else 'None'})")
                st.write(f"- Categorical columns: {len(categorical_cols)} ({', '.join(categorical_cols) if categorical_cols else 'None'})")
                st.write(f"- Date columns: {len(date_cols)} ({', '.join(date_cols) if date_cols else 'None'})")
        else:
            st.success(f"✅ Generated {visualizations_created} automatic visualization(s) for your query results!")

    except Exception as e:
        st.error(f"Error in auto-visualization: {e}")
        st.info("💡 The data table is available below the visualizations section.")


def show_example_queries():
    """Show example queries that users can try."""
    st.subheader("💡 Example Queries")
    
    examples = [
        ("Show dormant accounts", "Show me all accounts that are expected to be dormant"),
        ("Account types breakdown", "What are the different account types and how many of each?"),
        ("High balance accounts", "Find accounts with balance over 10000"),
        ("Article 3 process", "Show accounts that require Article 3 process"),
        ("Recent activity", "Show accounts with activity in the last 30 days")
    ]
    
    st.write("Click any example to try it:")
    
    cols = st.columns(len(examples))
    for i, (title, query) in enumerate(examples):
        with cols[i]:
            if st.button(title, key=f"example_{i}", use_container_width=True):
                # Set the query in the text area
                st.session_state['sql_bot_nl_query_input'] = query
                st.rerun()


# Quick examples for users
def show_database_stats():
    """Show quick database statistics."""
    st.subheader("📊 Database Overview")
    
    try:
        conn = get_db_connection()
        if conn:
            # Get table counts
            schema = get_db_schema()
            if schema:
                st.write(f"**Available Tables:** {len(schema)}")
                
                # Show table sizes
                for table_name in schema.keys():
                    try:
                        if hasattr(conn, 'execute'):
                            with conn.connect() as test_conn:
                                count_result = test_conn.execute(f"SELECT COUNT(*) FROM [{table_name}]")
                                row_count = count_result.fetchone()[0]
                        else:
                            cursor = conn.cursor()
                            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
                            row_count = cursor.fetchone()[0]
                            cursor.close()
                        
                        st.write(f"- **{table_name}:** {row_count:,} rows")
                        
                    except Exception as e:
                        st.write(f"- **{table_name}:** Error getting count")
    except Exception as e:
        st.error(f"Error getting database stats: {e}")
# ADD THESE FUNCTIONS TO ui/sqlbot_ui.py

def enhanced_generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode=False):
    """
    Enhanced SQL generation with datetime-aware prompting.
    """
    try:
        # Enhanced prompt template with datetime awareness
        enhanced_prompt_template = create_datetime_aware_prompt()
        
        nl_to_sql_prompt = PromptTemplate.from_template(enhanced_prompt_template)
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


def create_datetime_aware_prompt():
    """
    Create an enhanced SQL generation prompt that's aware of actual datetime column types.
    """
    # Get datetime column information if available
    datetime_info = st.session_state.get('datetime_columns_info', {})
    
    datetime_guidance = """
IMPORTANT DATETIME GUIDELINES FOR YOUR DATABASE:
- DATE columns (store only dates): Use 'YYYY-MM-DD' format
- DATETIME2 columns (store date + time): Use 'YYYY-MM-DD HH:MM:SS' format
- Always use proper date functions for comparisons

SPECIFIC DATE COLUMN INFORMATION:
"""
    
    if datetime_info:
        for table_name, columns in datetime_info.items():
            if columns:
                datetime_guidance += f"\nTable {table_name}:\n"
                for col_info in columns:
                    col_name = col_info['column']
                    col_type = col_info['formatted_type']
                    datetime_guidance += f"  - {col_name}: {col_type}\n"
                    
                    # Add specific guidance based on type
                    if col_info['type'] == 'date':
                        datetime_guidance += f"    Use: WHERE {col_name} >= '2024-01-01'\n"
                    elif col_info['type'] == 'datetime2':
                        datetime_guidance += f"    Use: WHERE {col_name} >= '2024-01-01 00:00:00'\n"
    
    datetime_guidance += """
COMMON DATE QUERY PATTERNS:
- Last 30 days: WHERE date_column >= DATEADD(DAY, -30, GETDATE())
- This year: WHERE YEAR(date_column) = YEAR(GETDATE())
- Date range: WHERE date_column BETWEEN '2024-01-01' AND '2024-12-31'
- Today: WHERE CAST(date_column AS DATE) = CAST(GETDATE() AS DATE)
"""
    
    enhanced_prompt = f"""
You are an expert SQL query generator for SQL Server/Azure SQL. Convert the natural language question to a syntactically correct SQL query.

{datetime_guidance}

Database Schema:
{{schema}}

Question: {{question}}

Generate only the SQL query without any explanation or markdown formatting.
Make sure to use the correct date formats and functions based on the column types shown above.
"""
    
    return enhanced_prompt


def enhanced_execute_sql_query(sql_query, conn):
    """
    Enhanced SQL query execution with better datetime handling.
    """
    try:
        cleaned_query = clean_sql_query(sql_query)
        
        st.info(f"🔍 **Executing SQL Query**")
        st.code(cleaned_query, language='sql')

        with st.spinner("⏳ Executing query..."):
            # Enhanced pandas.read_sql with datetime parsing
            try:
                # Try to identify datetime columns for parsing
                datetime_columns = identify_datetime_columns_for_query(cleaned_query)
                
                if datetime_columns:
                    results_df = pd.read_sql(
                        cleaned_query, 
                        conn,
                        parse_dates=datetime_columns,
                        dtype_backend='numpy_nullable'
                    )
                else:
                    results_df = pd.read_sql(
                        cleaned_query, 
                        conn,
                        parse_dates=True,
                        dtype_backend='numpy_nullable'
                    )
            except Exception as parse_error:
                # Fallback to basic read_sql
                st.warning(f"Advanced datetime parsing failed, using basic method: {parse_error}")
                results_df = pd.read_sql(cleaned_query, conn)
            
            # Post-process datetime columns that might have been missed
            results_df = fix_datetime_columns(results_df)

        # Display results with enhanced datetime formatting
        st.subheader("Query Results")
        if not results_df.empty:
            # Show basic statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Returned", len(results_df))
            with col2:
                st.metric("Columns", len(results_df.columns))
            with col3:
                memory_mb = results_df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            
            # Enhanced auto-generate visualizations
            auto_generate_visualizations_enhanced(results_df)
            
            # Display the data table with better datetime formatting
            st.subheader("📊 Data Table")
            formatted_df = format_datetime_for_display(results_df)
            st.dataframe(formatted_df, use_container_width=True, height=400)
            
            # Show enhanced column information
            show_enhanced_column_info(results_df)

            # Add download button
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"sql_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_results_csv"
            )

            add_query_to_history(cleaned_query, results_df)
        else:
            st.info("✅ Query executed successfully but returned no results.")
            add_query_to_history(cleaned_query, results_df)

    except Exception as e:
        st.error(f"❌ Query execution error: {e}")
        
        # Enhanced error guidance for datetime issues
        error_str = str(e).lower()
        if "conversion failed when converting date" in error_str:
            st.info("💡 **DateTime Error:** Try using explicit date formats like 'YYYY-MM-DD' or use CONVERT/CAST functions.")
            st.code("-- Example fixes:\nSELECT * FROM table WHERE date_column >= '2024-01-01'\n-- Or:\nSELECT * FROM table WHERE CAST(date_column AS DATE) = '2024-01-01'", language='sql')
        elif "invalid column name" in error_str:
            st.info("💡 **Tip:** Check if all column names exist in the database schema above.")
        elif "invalid object name" in error_str:
            st.info("💡 **Tip:** Check if the table name is correct. Available tables are shown in the schema.")
        elif "syntax error" in error_str:
            st.info("💡 **Tip:** There's a SQL syntax error. Try regenerating the query or check the SQL manually.")
        
        add_query_to_history(sql_query, None, error=str(e))


def identify_datetime_columns_for_query(sql_query):
    """
    Try to identify which columns in the query might be datetime columns.
    """
    datetime_info = st.session_state.get('datetime_columns_info', {})
    datetime_cols = []
    
    for table_name, columns in datetime_info.items():
        for col_info in columns:
            col_name = col_info['column']
            if col_name.lower() in sql_query.lower():
                datetime_cols.append(col_name)
    
    return datetime_cols if datetime_cols else None


def fix_datetime_columns(df):
    """Post-process DataFrame to fix datetime columns that pandas missed."""
    import re
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to detect if this is actually a datetime column
            if is_datetime_column(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass  # If conversion fails, leave as is
    return df


def is_datetime_column(series):
    """Heuristic to detect if an object column contains datetime values."""
    if series.empty:
        return False
    
    # Sample first few non-null values
    sample = series.dropna().head(10)
    if sample.empty:
        return False
    
    datetime_count = 0
    for value in sample:
        if isinstance(value, str):
            # Common datetime patterns
            datetime_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            ]
            
            for pattern in datetime_patterns:
                if re.search(pattern, str(value)):
                    datetime_count += 1
                    break
    
    # If more than 50% look like datetimes, treat as datetime
    return datetime_count / len(sample) > 0.5


def format_datetime_for_display(df):
    """Format datetime columns for better display in Streamlit."""
    formatted_df = df.copy()
    
    for col in formatted_df.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted_df[col]):
            # Format datetime for display
            formatted_df[col] = formatted_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Replace NaT with empty string
            formatted_df[col] = formatted_df[col].fillna('')
        elif pd.api.types.is_timedelta64_dtype(formatted_df[col]):
            # Format timedelta for display
            formatted_df[col] = formatted_df[col].astype(str)
            formatted_df[col] = formatted_df[col].replace('NaT', '')
    
    return formatted_df


def show_enhanced_column_info(df):
    """Show enhanced column information including datetime details."""
    with st.expander("📋 Enhanced Column Information"):
        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].count()
            null_count = len(df) - non_null
            unique_count = df[col].nunique()
            
            # Add datetime-specific info
            extra_info = ""
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if non_null > 0:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    extra_info = f"Range: {min_date} to {max_date}"
            elif pd.api.types.is_numeric_dtype(df[col]) and non_null > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                extra_info = f"Range: {min_val} to {max_val}"
            
            col_info.append({
                'Column': col,
                'Data Type': dtype,
                'Non-Null': non_null,
                'Null': null_count,
                'Unique Values': unique_count,
                'Additional Info': extra_info
            })
        
        col_info_df = pd.DataFrame(col_info)
        st.dataframe(col_info_df, use_container_width=True)


def auto_generate_visualizations_enhanced(results_df):
    """Enhanced auto-visualization with better datetime support."""
    if results_df.empty or len(results_df) == 0:
        return
    
    st.subheader("📈 Auto-Generated Visualizations")
    
    # Analyze the data structure with enhanced datetime detection
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Enhanced datetime detection
    date_cols = []
    for col in results_df.columns:
        if pd.api.types.is_datetime64_any_dtype(results_df[col]):
            date_cols.append(col)
    
    # Filter categorical columns
    categorical_cols = [col for col in categorical_cols 
                       if not col.lower().endswith('_id') 
                       and not col.lower().startswith('id')
                       and results_df[col].nunique() <= 20
                       and results_df[col].nunique() > 1
                       and col not in date_cols]
    
    visualizations_created = 0
    max_visualizations = 3
    
    try:
        # Enhanced time series visualization
        if date_cols and len(results_df) > 1 and visualizations_created < max_visualizations:
            date_col = date_cols[0]
            
            try:
                # Sort by date
                df_sorted = results_df.sort_values(date_col)
                
                if numeric_cols:
                    # Time series with numeric value
                    num_col = numeric_cols[0]
                    fig = px.line(
                        df_sorted, 
                        x=date_col, 
                        y=num_col,
                        title=f"{num_col.replace('_', ' ').title()} Over Time"
                    )
                    fig.update_xaxes(title=date_col.replace('_', ' ').title())
                    fig.update_yaxes(title=num_col.replace('_', ' ').title())
                else:
                    # Count occurrences over time
                    date_counts = df_sorted.groupby(df_sorted[date_col].dt.date).size().reset_index()
                    date_counts.columns = ['date', 'count']
                    fig = px.line(
                        date_counts, 
                        x='date', 
                        y='count',
                        title=f"Activity Over Time"
                    )
                
                st.plotly_chart(fig, use_container_width=True, key=f"timeseries_{date_col}")
                visualizations_created += 1
                st.caption(f"📊 Time series chart showing trends over '{date_col}'")
            except Exception as e:
                st.write(f"Could not create time series: {e}")

        # Continue with other visualization types from the original function...
        # [Include the rest of the original auto_generate_visualizations logic here]
        
    except Exception as e:
        st.error(f"Error in auto-visualization: {e}")


def show_datetime_examples():
    """Show specific examples for datetime queries."""
    st.subheader("💡 DateTime Query Examples")
    
    examples = [
        ("Recent records", "Show me records from the last 30 days"),
        ("This year's data", "Show all accounts created this year"),
        ("Date range", "Show accounts created between 2023-01-01 and 2023-12-31"),
        ("Monthly summary", "Show count of accounts created by month"),
        ("Activity today", "Show accounts with activity today")
    ]
    
    st.write("**Click any datetime example to try it:**")
    
    cols = st.columns(len(examples))
    for i, (title, query) in enumerate(examples):
        with cols[i]:
            if st.button(title, key=f"datetime_example_{i}", use_container_width=True):
                st.session_state['sql_bot_nl_query_input'] = query
                st.rerun()

# Add this to show database stats
if __name__ == "__main__":
    show_database_stats()
