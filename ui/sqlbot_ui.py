import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import traceback
from typing import List, Dict, Optional, Tuple, Any

# Import necessary modules
from database.connection import get_db_connection
from database.schema import get_db_schema, get_datetime_columns_info, is_datetime_column_in_table
from ai.llm import load_llm  # This is the correct function name in your ai/llm.py
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from ai.visualization import generate_plot, create_insights_chart

def init_sql_bot_session_state():
    """Initialize session state variables for SQL Bot."""
    if 'sql_bot_queries_history' not in st.session_state:
        st.session_state['sql_bot_queries_history'] = []
    if 'sql_bot_nl_query_input' not in st.session_state:
        st.session_state['sql_bot_nl_query_input'] = ""
    if 'sql_bot_generated_query' not in st.session_state:
        st.session_state['sql_bot_generated_query'] = ""
    if 'sql_bot_advanced_mode' not in st.session_state:
        st.session_state['sql_bot_advanced_mode'] = False
    if 'datetime_columns_info' not in st.session_state:
        st.session_state['datetime_columns_info'] = {}


def get_schema_text(schema_dict):
    """Convert schema dictionary to formatted text for LLM."""
    if not schema_dict:
        return "No schema available."

    schema_text = "Database Schema:\n\n"
    for table_name, columns in schema_dict.items():
        schema_text += f"Table: {table_name}\n"
        for column_name, column_type in columns:
            schema_text += f"  - {column_name}: {column_type}\n"
        schema_text += "\n"

    # Add datetime column information if available
    datetime_info = get_datetime_columns_info()
    if datetime_info:
        schema_text += "\nDATETIME COLUMN DETAILS:\n"
        for table_name, columns in datetime_info.items():
            if columns:
                schema_text += f"Table {table_name} datetime columns:\n"
                for col_info in columns:
                    schema_text += f"  - {col_info['column']}: {col_info['formatted_type']} (SQL Server {col_info['type']})\n"
                schema_text += "\n"

    return schema_text


def create_datetime_aware_prompt():
    """
    Create an enhanced SQL generation prompt that's aware of actual datetime column types.
    This is the KEY enhancement for better datetime SQL generation.
    """
    # Get datetime column information if available
    datetime_info = get_datetime_columns_info()

    datetime_guidance = """
IMPORTANT DATETIME GUIDELINES FOR YOUR DATABASE:
- DATE columns (store only dates): Use 'YYYY-MM-DD' format
- DATETIME2 columns (store date + time): Use 'YYYY-MM-DD HH:MM:SS' format
- Always use proper SQL Server date functions for comparisons

SPECIFIC DATE COLUMN INFORMATION:
"""

    if datetime_info:
        for table_name, columns in datetime_info.items():
            if columns:
                datetime_guidance += f"\nTable {table_name}:\n"
                for col_info in columns:
                    col_name = col_info['column']
                    col_type = col_info['formatted_type']
                    raw_type = col_info['type']
                    datetime_guidance += f"  - {col_name}: {col_type}\n"

                    # Add specific guidance based on type
                    if raw_type == 'date':
                        datetime_guidance += f"    Example: WHERE {col_name} >= '2024-01-01'\n"
                        datetime_guidance += f"    Example: WHERE {col_name} BETWEEN '2024-01-01' AND '2024-12-31'\n"
                    elif raw_type in ['datetime2', 'datetime']:
                        datetime_guidance += f"    Example: WHERE {col_name} >= '2024-01-01 00:00:00'\n"
                        datetime_guidance += f"    Example: WHERE {col_name} >= DATEADD(DAY, -30, GETDATE())\n"

    datetime_guidance += """

COMMON DATE QUERY PATTERNS (Use these for temporal queries):
- Last 30 days: WHERE date_column >= DATEADD(DAY, -30, GETDATE())
- This year: WHERE YEAR(date_column) = YEAR(GETDATE())
- Last year: WHERE YEAR(date_column) = YEAR(GETDATE()) - 1
- Date range: WHERE date_column BETWEEN '2024-01-01' AND '2024-12-31'
- Today: WHERE CAST(date_column AS DATE) = CAST(GETDATE() AS DATE)
- This month: WHERE YEAR(date_column) = YEAR(GETDATE()) AND MONTH(date_column) = MONTH(GETDATE())
- Monthly grouping: GROUP BY YEAR(date_column), MONTH(date_column)
- Recent activity: WHERE date_column >= DATEADD(DAY, -7, GETDATE())

TEMPORAL KEYWORDS MAPPING:
- "recent", "recently" → Use DATEADD(DAY, -30, GETDATE()) or DATEADD(DAY, -7, GETDATE())
- "this year" → Use YEAR(date_column) = YEAR(GETDATE())
- "last year" → Use YEAR(date_column) = YEAR(GETDATE()) - 1
- "this month" → Use YEAR/MONTH functions with GETDATE()
- "today" → Use CAST(date_column AS DATE) = CAST(GETDATE() AS DATE)
- "last 30 days" → Use DATEADD(DAY, -30, GETDATE())
"""

    enhanced_prompt = f"""
You are an expert SQL query generator for SQL Server/Azure SQL Database. Convert the natural language question to a syntactically correct SQL query.

{datetime_guidance}

Database Schema:
{{schema}}

Question: {{question}}

CRITICAL INSTRUCTIONS:
1. Use the EXACT column names and types shown in the schema above
2. For DATE columns, use 'YYYY-MM-DD' format
3. For DATETIME2 columns, use 'YYYY-MM-DD HH:MM:SS' format or SQL Server date functions
4. Always use SQL Server compatible functions (DATEADD, YEAR, MONTH, GETDATE)
5. Pay attention to the specific datetime types for each column
6. Use proper SQL Server syntax for all operations

Generate ONLY the SQL query without any explanation, markdown formatting, or extra text.
"""

    return enhanced_prompt


def enhanced_generate_sql_from_nl(nl_query: str, schema_text: str, llm, is_advanced_mode: bool = False) -> Optional[
    str]:
    """
    Enhanced SQL generation with datetime-aware prompting.
    This replaces the original generate_sql_from_nl function.
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


def identify_datetime_columns_for_query(sql_query: str) -> Optional[List[str]]:
    """
    Try to identify which columns in the query might be datetime columns.
    This helps pandas parse them correctly.
    """
    datetime_info = get_datetime_columns_info()
    datetime_cols = []

    # Convert query to lowercase for case-insensitive matching
    query_lower = sql_query.lower()

    for table_name, columns in datetime_info.items():
        for col_info in columns:
            col_name = col_info['column']
            # Check if column name appears in the query
            if col_name.lower() in query_lower:
                datetime_cols.append(col_name)

    return datetime_cols if datetime_cols else None


def fix_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process DataFrame to fix datetime columns that pandas missed.
    This handles cases where datetime columns are returned as objects.
    """
    import re

    for col in df.columns:
        if df[col].dtype == 'object' and not df[col].empty:
            # Try to detect if this is actually a datetime column
            if is_datetime_column_heuristic(df[col]):
                try:
                    # Try different datetime parsing methods
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                except Exception:
                    try:
                        # Try with specific formats
                        df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m-%d')
                    except Exception:
                        pass  # If all conversions fail, leave as is
    return df


def is_datetime_column_heuristic(series: pd.Series) -> bool:
    """
    Heuristic to detect if an object column contains datetime values.
    """
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
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            ]

            for pattern in datetime_patterns:
                if re.search(pattern, str(value)):
                    datetime_count += 1
                    break

    # If more than 50% look like datetimes, treat as datetime
    return datetime_count / len(sample) > 0.5


def format_datetime_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format datetime columns for better display in Streamlit.
    """
    formatted_df = df.copy()

    for col in formatted_df.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted_df[col]):
            # Format datetime for display
            try:
                # Check if it has time component
                has_time = formatted_df[col].dt.time.apply(lambda x: x != pd.Timestamp('00:00:00').time()).any()

                if has_time:
                    formatted_df[col] = formatted_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    formatted_df[col] = formatted_df[col].dt.strftime('%Y-%m-%d')

                # Replace NaT with empty string
                formatted_df[col] = formatted_df[col].fillna('')
            except Exception:
                # If formatting fails, convert to string
                formatted_df[col] = formatted_df[col].astype(str).replace('NaT', '')

        elif pd.api.types.is_timedelta64_dtype(formatted_df[col]):
            # Format timedelta for display
            formatted_df[col] = formatted_df[col].astype(str).replace('NaT', '')

    return formatted_df


def show_enhanced_column_info(df: pd.DataFrame):
    """
    Show enhanced column information including datetime details.
    """
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
                    try:
                        min_date = df[col].min()
                        max_date = df[col].max()
                        date_range = max_date - min_date
                        extra_info = f"Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range.days} days)"
                    except Exception:
                        extra_info = "Date range calculation failed"
            elif pd.api.types.is_numeric_dtype(df[col]) and non_null > 0:
                try:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    extra_info = f"Range: {min_val:,.2f} to {max_val:,.2f}"
                except Exception:
                    extra_info = "Numeric range calculation failed"
            elif dtype == 'object' and non_null > 0:
                try:
                    avg_length = df[col].str.len().mean()
                    extra_info = f"Avg text length: {avg_length:.1f} chars"
                except Exception:
                    extra_info = "Text analysis failed"

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


def enhanced_execute_sql_query(sql_query: str, conn):
    """
    Enhanced SQL query execution with better datetime handling and visualization.py integration.
    This replaces the original execute_sql_query function.
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
                    st.info(f"🕒 Detected datetime columns: {', '.join(datetime_columns)}")
                    results_df = pd.read_sql(
                        cleaned_query,
                        conn,
                        parse_dates=datetime_columns,
                        dtype_backend='numpy_nullable'
                    )
                else:
                    # Try automatic datetime parsing
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
        st.subheader("📊 Query Results")
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

            # Enhanced auto-generate visualizations using visualization.py
            auto_generate_visualizations_enhanced(results_df)

            # Display the data table with better datetime formatting
            st.subheader("📋 Data Table")
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
            st.info(
                "💡 **DateTime Error:** Try using explicit date formats like 'YYYY-MM-DD' or use CONVERT/CAST functions.")
            st.code(
                "-- Example fixes:\nSELECT * FROM table WHERE date_column >= '2024-01-01'\n-- Or:\nSELECT * FROM table WHERE CAST(date_column AS DATE) = '2024-01-01'",
                language='sql')
        elif "invalid column name" in error_str:
            st.info("💡 **Tip:** Check if all column names exist in the database schema above.")
        elif "invalid object name" in error_str:
            st.info("💡 **Tip:** Check if the table name is correct. Available tables are shown in the schema.")
        elif "syntax error" in error_str:
            st.info("💡 **Tip:** There's a SQL syntax error. Try regenerating the query or check the SQL manually.")

        add_query_to_history(sql_query, None, error=str(e))


def auto_generate_visualizations_enhanced(results_df: pd.DataFrame):
    """
    Enhanced auto-visualization using functions from visualization.py
    """
    if results_df is None or results_df.empty or len(results_df) == 0:
        st.info("💡 No data available for visualization.")
        return

    try:
        st.subheader("📈 Auto-Generated Visualizations")

        # Analyze the data structure
        numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = []
        
        # Enhanced datetime detection
        for col in results_df.columns:
            if pd.api.types.is_datetime64_any_dtype(results_df[col]):
                date_cols.append(col)

        # Filter categorical columns (exclude ID columns and columns with too many unique values)
        categorical_cols = [col for col in categorical_cols
                            if not col.lower().endswith('_id')
                            and not col.lower().startswith('id')
                            and results_df[col].nunique() <= 20
                            and results_df[col].nunique() > 1
                            and col not in date_cols]

        visualizations_created = 0
        max_visualizations = 4

        # 1. Pie chart for categorical data (if suitable)
        if categorical_cols and visualizations_created < max_visualizations:
            cat_col = categorical_cols[0]
            unique_count = results_df[cat_col].nunique()
            
            if unique_count <= 10:  # Suitable for pie chart
                try:
                    chart = create_insights_chart(
                        data=results_df,
                        labels=cat_col,
                        chart_type='pie',
                        title=f"Distribution of {cat_col.replace('_', ' ').title()}"
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"pie_{cat_col}")
                        visualizations_created += 1
                        st.caption(f"📊 Pie chart showing distribution of '{cat_col.replace('_', ' ')}'")
                except Exception as e:
                    st.warning(f"Could not create pie chart: {str(e)}")

        # 2. Bar chart for categorical data
        if categorical_cols and visualizations_created < max_visualizations:
            cat_col = categorical_cols[0] if visualizations_created == 0 else (
                categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
            )
            
            try:
                chart = create_insights_chart(
                    data=results_df,
                    labels=cat_col,
                    chart_type='bar',
                    title=f"Count by {cat_col.replace('_', ' ').title()}"
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"bar_{cat_col}")
                    visualizations_created += 1
                    st.caption(f"📊 Bar chart showing counts for '{cat_col.replace('_', ' ')}'")
            except Exception as e:
                st.warning(f"Could not create bar chart: {str(e)}")

        # 3. Histogram for numeric data
        if numeric_cols and visualizations_created < max_visualizations:
            num_col = numeric_cols[0]
            
            try:
                chart = create_insights_chart(
                    data=results_df,
                    labels=num_col,
                    chart_type='histogram',
                    title=f"Distribution of {num_col.replace('_', ' ').title()}"
                )
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"histogram_{num_col}")
                    visualizations_created += 1
                    st.caption(f"📊 Histogram showing distribution of '{num_col.replace('_', ' ')}'")
            except Exception as e:
                st.warning(f"Could not create histogram: {str(e)}")

        # 4. Scatter plot for two numeric columns
        if len(numeric_cols) >= 2 and visualizations_created < max_visualizations:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            try:
                # Create LLM output format for generate_plot function
                plot_spec = {
                    "plot_type": "scatter",
                    "x_column": x_col,
                    "y_column": y_col,
                    "color_column": None,
                    "names_column": None,
                    "title": f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}"
                }
                
                chart, response_text = generate_plot(plot_spec, results_df)
                
                if chart:
                    st.plotly_chart(chart, use_container_width=True, key=f"scatter_{x_col}_{y_col}")
                    visualizations_created += 1
                    st.caption(f"📊 {response_text}")
            except Exception as e:
                st.warning(f"Could not create scatter plot: {str(e)}")

        # 5. Box plot for numeric data with categorical grouping
        if numeric_cols and categorical_cols and visualizations_created < max_visualizations:
            num_col = numeric_cols[0]
            cat_col = categorical_cols[0]
            
            # Only create box plot if categorical column has reasonable number of categories
            if results_df[cat_col].nunique() <= 10:
                try:
                    plot_spec = {
                        "plot_type": "box",
                        "x_column": cat_col,
                        "y_column": num_col,
                        "color_column": None,
                        "names_column": None,
                        "title": f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
                    }
                    
                    chart, response_text = generate_plot(plot_spec, results_df)
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"box_{num_col}_{cat_col}")
                        visualizations_created += 1
                        st.caption(f"📊 {response_text}")
                except Exception as e:
                    st.warning(f"Could not create box plot: {str(e)}")

        # 6. Time series visualization for datetime columns
        if date_cols and visualizations_created < max_visualizations:
            date_col = date_cols[0]
            
            try:
                if numeric_cols:
                    # Time series with numeric value
                    num_col = numeric_cols[0]
                    plot_spec = {
                        "plot_type": "scatter",
                        "x_column": date_col,
                        "y_column": num_col,
                        "color_column": None,
                        "names_column": None,
                        "title": f"{num_col.replace('_', ' ').title()} Over Time"
                    }
                    
                    chart, response_text = generate_plot(plot_spec, results_df)
                    
                    if chart:
                        # Convert scatter to line for time series
                        chart.update_traces(mode='lines+markers')
                        st.plotly_chart(chart, use_container_width=True, key=f"timeseries_{date_col}_{num_col}")
                        visualizations_created += 1
                        st.caption(f"📊 Time series: {response_text}")
                else:
                    # Count occurrences over time using bar chart
                    date_counts = results_df.groupby(results_df[date_col].dt.date).size().reset_index()
                    date_counts.columns = [date_col, 'count']
                    
                    chart = create_insights_chart(
                        data=date_counts,
                        labels=date_col,
                        values='count',
                        chart_type='bar',
                        title=f"Activity Count Over Time"
                    )
                    
                    if chart:
                        st.plotly_chart(chart, use_container_width=True, key=f"timeseries_count_{date_col}")
                        visualizations_created += 1
                        st.caption(f"📊 Time series showing activity count over '{date_col.replace('_', ' ')}'")
            except Exception as e:
                st.warning(f"Could not create time series visualization: {str(e)}")

        # Summary statistics for numeric columns
        if numeric_cols:
            try:
                st.subheader("📊 Summary Statistics")
                summary_stats = results_df[numeric_cols].describe()
                st.dataframe(summary_stats, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create summary statistics: {str(e)}")

        if visualizations_created == 0:
            st.info(
                "💡 No suitable columns found for automatic visualization. Try queries that return numeric data, dates, or categorical data with reasonable cardinality.")

    except Exception as e:
        st.error(f"❌ Error in auto-visualization: {str(e)}")
        st.info("💡 You can still view your data in the table below.")


def show_datetime_examples():
    """
    Show specific examples for datetime queries with click-to-try functionality.
    """
    st.subheader("🕒 DateTime Query Examples")

    examples = [
        ("Recent Activity", "Show me accounts with activity in the last 30 days", "recent"),
        ("This Year", "Show all accounts created this year", "this_year"),
        ("Date Range", "Show accounts created between 2020-01-01 and 2023-12-31", "date_range"),
        ("Monthly Trends", "Show count of accounts created by month", "monthly"),
        ("Dormant Analysis", "Show accounts with no activity since 2021", "dormant")
    ]

    st.write("**Click any example to try it:**")

    # Create columns for better layout
    cols = st.columns(len(examples))
    for i, (title, query, key) in enumerate(examples):
        with cols[i]:
            if st.button(
                    title,
                    key=f"datetime_example_{key}",
                    use_container_width=True,
                    help=f"Try: {query}"
            ):
                st.session_state['sql_bot_nl_query_input'] = query
                st.rerun()

    # Additional datetime tips
    with st.expander("💡 DateTime Query Tips"):
        st.write("""
        **Effective DateTime Query Patterns:**

        - **"Recent" queries**: Use terms like "last 30 days", "recent activity", "recently created"
        - **Year-based**: "this year", "2023", "last year" 
        - **Specific dates**: "after 2020-01-01", "before 2022-12-31"
        - **Time periods**: "last quarter", "this month", "past week"
        - **Comparisons**: "older than", "newer than", "between dates"

        **Example Natural Language Queries:**
        - "Show dormant accounts that haven't been active since 2021"
        - "Find accounts created this year with balance over 10000"
        - "Show monthly account creation trends"
        - "List accounts with recent activity but low balance"
        """)


# Replace the original functions with enhanced versions
def generate_sql_from_nl(nl_query: str, schema_text: str, llm, is_advanced_mode: bool = False) -> Optional[str]:
    """Enhanced SQL generation - this replaces the original function."""
    return enhanced_generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode)


def execute_sql_query(sql_query: str, conn):
    """Enhanced SQL execution - this replaces the original function."""
    return enhanced_execute_sql_query(sql_query, conn)


# Keep all existing utility functions (these don't need changes)
def clean_sql_query(sql_query_raw: str) -> str:
    """Clean and format the generated SQL query."""
    # Remove markdown code blocks if present
    sql_query = re.sub(r'```sql\n?', '', sql_query_raw)
    sql_query = re.sub(r'```\n?', '', sql_query)

    # Remove common prefixes
    sql_query = re.sub(r'^(Here\'s the SQL query:?\s*)', '', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'^(SQL:?\s*)', '', sql_query, flags=re.IGNORECASE)

    # Clean up whitespace
    sql_query = sql_query.strip()

    # Remove trailing semicolons (they can cause issues with some drivers)
    sql_query = sql_query.rstrip(';')

    return sql_query


def get_fallback_response(error_type: str) -> str:
    """Provide fallback SQL queries for common scenarios."""
    fallback_queries = {
        "sql_generation": "SELECT TOP 10 * FROM accounts_data",
        "connection_error": "SELECT 'Database connection unavailable' as message",
        "schema_error": "SELECT 'Schema information unavailable' as message"
    }
    return fallback_queries.get(error_type, "SELECT 1 as status")


def add_query_to_history(sql_query: str, results_df: Optional[pd.DataFrame], error: Optional[str] = None):
    """Add query to session history."""
    if 'sql_bot_queries_history' not in st.session_state:
        st.session_state['sql_bot_queries_history'] = []

    history_entry = {
        'timestamp': datetime.now(),
        'sql_query': sql_query,
        'row_count': len(results_df) if results_df is not None else 0,
        'success': error is None,
        'error': error
    }

    st.session_state['sql_bot_queries_history'].append(history_entry)

    # Keep only last 50 queries to prevent memory issues
    if len(st.session_state['sql_bot_queries_history']) > 50:
        st.session_state['sql_bot_queries_history'] = st.session_state['sql_bot_queries_history'][-50:]


def show_example_queries():
    """Show example queries that users can try."""
    st.subheader("💡 Example Queries")

    examples = [
        ("Basic Query", "Show me all accounts", "basic"),
        ("Filter Query", "Show accounts with balance over 10000", "filter"),
        ("Count Query", "How many accounts do we have?", "count"),
        ("Group Query", "Show account count by type", "group"),
        ("Advanced Query", "Show customers with multiple accounts", "advanced")
    ]

    st.write("**Click any example to try it:**")

    # Create columns for better layout
    cols = st.columns(len(examples))
    for i, (title, query, key) in enumerate(examples):
        with cols[i]:
            if st.button(
                    title,
                    key=f"example_{key}",
                    use_container_width=True,
                    help=f"Try: {query}"
            ):
                st.session_state['sql_bot_nl_query_input'] = query
                st.rerun()

    # Add datetime examples
    show_datetime_examples()


def show_query_history():
    """Display query history without nested expanders."""
    try:
        conn = get_db_connection()
        if not conn:
            st.write("No database connection available for query history.")
            return

        # Get query history
        history_query = "SELECT TOP 10 query_text, execution_time, created_at FROM sql_query_history ORDER BY created_at DESC"

        try:
            if hasattr(conn, 'connect'):
                with conn.connect() as test_conn:
                    df_history = pd.read_sql(history_query, test_conn)
            else:
                df_history = pd.read_sql(history_query, conn)

            if not df_history.empty:
                st.write("**📜 Recent Queries:**")

                # Display as simple table instead of expander
                for idx, row in df_history.iterrows():
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        # Truncate long queries for display
                        display_query = row['query_text'][:100] + "..." if len(row['query_text']) > 100 else row[
                            'query_text']
                        st.code(display_query, language="sql")

                    with col2:
                        if pd.notna(row['created_at']):
                            st.write(f"🕒 {row['created_at']}")
                        if pd.notna(row['execution_time']):
                            st.write(f"⏱️ {row['execution_time']:.2f}s")

                    # Add a button to reuse the query (without expander)
                    if st.button(f"📋 Use Query #{idx + 1}", key=f"history_btn_{idx}"):
                        st.session_state['sql_bot_generated_query'] = row['query_text']
                        st.success("Query loaded! Scroll up to see it.")
                        st.rerun()

                    st.divider()  # Add visual separation
            else:
                st.info("No query history available yet.")

        except Exception as e:
            st.write(f"Unable to load query history: {e}")

    except Exception as e:
        st.write(f"Query history unavailable: {e}")

def validate_sql_query(sql_query: str) -> Tuple[bool, str]:
    """Basic SQL query validation."""
    if not sql_query or not sql_query.strip():
        return False, "Query is empty"

    # Remove comments and extra whitespace
    cleaned_query = re.sub(r'--.*?\n', '', sql_query)
    cleaned_query = cleaned_query.strip()

    if not cleaned_query.lower().startswith('select'):
        return False, "Query must start with SELECT"

    # Basic syntax checks
    dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
    query_lower = cleaned_query.lower()

    for keyword in dangerous_keywords:
        if keyword in query_lower:
            return False, f"Query contains potentially dangerous keyword: {keyword.upper()}"

    # Check for balanced parentheses
    if cleaned_query.count('(') != cleaned_query.count(')'):
        return False, "Unbalanced parentheses in query"

    return True, "Query validation passed"


def show_advanced_options():
    """Show advanced SQL Bot options."""
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)

        with col1:
            st.session_state['sql_bot_advanced_mode'] = st.checkbox(
                "Advanced Mode",
                value=st.session_state.get('sql_bot_advanced_mode', False),
                help="Enable more complex query generation and additional features"
            )

            show_query_validation = st.checkbox(
                "Validate Queries",
                value=True,
                help="Perform basic validation on generated SQL queries"
            )

        with col2:
            auto_execute = st.checkbox(
                "Auto-execute generated queries",
                value=False,
                help="Automatically execute queries after generation (use with caution)"
            )

            show_execution_plan = st.checkbox(
                "Show execution details",
                value=False,
                help="Display query execution time and other details"
            )

        return {
            'advanced_mode': st.session_state['sql_bot_advanced_mode'],
            'validate_queries': show_query_validation,
            'auto_execute': auto_execute,
            'show_execution_plan': show_execution_plan
        }


def show_schema_info():
    """Display database schema information."""
    st.subheader("🗂️ Database Schema")

    schema = get_db_schema()

    if schema:
        # Show summary statistics
        total_tables = len(schema)
        total_columns = sum(len(columns) for columns in schema.values())

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Tables", total_tables)
        with col2:
            st.metric("Total Columns", total_columns)

        # Show datetime column info if available
        datetime_info = get_datetime_columns_info()
        if datetime_info:
            st.write("**🕒 DateTime Columns Detected:**")
            for table_name, columns in datetime_info.items():
                if columns:
                    st.write(f"**{table_name}:**")
                    for col_info in columns:
                        st.write(f"  - `{col_info['column']}`: {col_info['formatted_type']}")

        # Show detailed schema using separate expanders (not nested)
        st.write("**📋 Detailed Schema:**")
        for table_name, columns in schema.items():
            # Create individual expanders for each table (not nested)
            with st.expander(f"Table: {table_name} ({len(columns)} columns)"):
                for column_name, column_type in columns:
                    # Highlight datetime columns
                    is_datetime, _ = is_datetime_column_in_table(table_name, column_name)
                    if is_datetime:
                        st.write(f"🕒 **{column_name}**: {column_type}")
                    else:
                        st.write(f"  - {column_name}: {column_type}")
    else:
        st.error("Could not retrieve database schema.")

def main_sql_bot_interface():
    """
    Main SQL Bot interface with enhanced datetime functionality.
    This is the main function that renders the SQL Bot UI.
    """
    st.title("🤖 SQL Bot with Enhanced DateTime Support")
    st.write("Ask questions about your data in natural language, and I'll generate SQL queries for you.")

    # Initialize session state
    init_sql_bot_session_state()

    # Show database schema information
    show_schema_info()

    # Get LLM instance
    try:
        llm = load_llm()  # Using the correct function name from your ai/llm.py
        if llm is None:
            st.error("❌ Could not initialize LLM. Please check your configuration.")
            return
    except Exception as e:
        st.error(f"❌ LLM initialization error: {e}")
        return

    # Get database schema
    schema = get_db_schema()
    schema_text = get_schema_text(schema)

    # Show advanced options
    advanced_options = show_advanced_options()

    # Main query interface
    st.subheader("💬 Ask Your Question")

    # Natural language input
    nl_query = st.text_area(
        "Enter your question about the data:",
        value=st.session_state.get('sql_bot_nl_query_input', ''),
        height=100,
        placeholder="e.g., 'Show me accounts created this year with balance over 10000'"
    )

    # Update session state
    st.session_state['sql_bot_nl_query_input'] = nl_query

    # Buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        generate_sql_button = st.button(
            "🔄 Generate SQL",
            type="primary",
            disabled=not nl_query.strip(),
            use_container_width=True
        )

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    with col3:
        example_button = st.button("💡 Examples", use_container_width=True)

    # Handle button clicks
    if clear_button:
        st.session_state['sql_bot_nl_query_input'] = ""
        st.session_state['sql_bot_generated_query'] = ""
        st.rerun()

    if example_button:
        st.session_state['show_examples'] = not st.session_state.get('show_examples', False)
        st.rerun()

    # Show examples if requested
    if st.session_state.get('show_examples', False):
        show_example_queries()

    # Generate SQL query
    if generate_sql_button and nl_query.strip():
        with st.spinner("🧠 Generating SQL query..."):
            try:
                generated_query = generate_sql_from_nl(
                    nl_query,
                    schema_text,
                    llm,
                    advanced_options['advanced_mode']
                )

                if generated_query:
                    st.session_state['sql_bot_generated_query'] = generated_query
                    st.success("✅ SQL query generated successfully!")
                else:
                    st.error("❌ Failed to generate SQL query.")

            except Exception as e:
                st.error(f"❌ Error generating SQL: {e}")

    # Display and execute generated query
    if st.session_state.get('sql_bot_generated_query'):
        st.subheader("🔍 Generated SQL Query")

        # Allow editing of the generated query
        edited_query = st.text_area(
            "Review and edit the generated SQL query:",
            value=st.session_state['sql_bot_generated_query'],
            height=150,
            key="sql_editor"
        )

        # Query validation
        if advanced_options['validate_queries']:
            is_valid, validation_message = validate_sql_query(edited_query)
            if is_valid:
                st.success(f"✅ {validation_message}")
            else:
                st.error(f"❌ {validation_message}")

        # Execute button
        col1, col2 = st.columns([3, 1])

        with col1:
            execute_button = st.button(
                "▶️ Execute Query",
                type="primary",
                disabled=advanced_options['validate_queries'] and not is_valid,
                use_container_width=True
            )

        with col2:
            save_query_button = st.button("💾 Save Query", use_container_width=True)

        # Auto-execute if enabled
        if advanced_options['auto_execute'] and edited_query != st.session_state['sql_bot_generated_query']:
            execute_button = True

        # Execute the query
        if execute_button:
            conn = get_db_connection()
            if conn:
                if advanced_options['show_execution_plan']:
                    start_time = datetime.now()

                execute_sql_query(edited_query, conn)

                if advanced_options['show_execution_plan']:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    st.info(f"⏱️ Query execution time: {execution_time:.2f} seconds")
            else:
                st.error("❌ No database connection available.")

        # Save query functionality
        if save_query_button:
            # Add to history manually
            add_query_to_history(edited_query, None)
            st.success("💾 Query saved to history!")

    # Show query history
    with st.expander("📜 Query History"):
        show_query_history()


def auto_generate_visualizations(results_df: pd.DataFrame):
    """
    Wrapper function for backward compatibility.
    This calls the enhanced visualization function.
    """
    return auto_generate_visualizations_enhanced(results_df)


# Additional utility functions for enhanced datetime support
def get_temporal_keywords():
    """Return a dictionary of temporal keywords and their SQL equivalents."""
    return {
        'recent': 'DATEADD(DAY, -30, GETDATE())',
        'recently': 'DATEADD(DAY, -7, GETDATE())',
        'this year': 'YEAR(date_column) = YEAR(GETDATE())',
        'last year': 'YEAR(date_column) = YEAR(GETDATE()) - 1',
        'this month': 'YEAR(date_column) = YEAR(GETDATE()) AND MONTH(date_column) = MONTH(GETDATE())',
        'today': 'CAST(date_column AS DATE) = CAST(GETDATE() AS DATE)',
        'yesterday': 'CAST(date_column AS DATE) = CAST(DATEADD(DAY, -1, GETDATE()) AS DATE)',
        'last 30 days': 'date_column >= DATEADD(DAY, -30, GETDATE())',
        'last week': 'date_column >= DATEADD(DAY, -7, GETDATE())',
        'last month': 'date_column >= DATEADD(MONTH, -1, GETDATE())'
    }


def suggest_datetime_improvements(nl_query: str) -> List[str]:
    """
    Analyze natural language query and suggest datetime-specific improvements.
    """
    suggestions = []
    query_lower = nl_query.lower()

    # Check for temporal keywords
    temporal_keywords = get_temporal_keywords()
    for keyword in temporal_keywords:
        if keyword in query_lower:
            suggestions.append(
                f"✨ Detected temporal keyword '{keyword}' - will use appropriate SQL Server date functions")

    # Check for specific date formats
    date_patterns = [
        (r'\d{4}-\d{2}-\d{2}', "ISO date format detected"),
        (r'\d{2}/\d{2}/\d{4}', "US date format detected - will convert to ISO format"),
        (r'\d{2}-\d{2}-\d{4}', "Alternative date format detected - will convert to ISO format")
    ]

    for pattern, message in date_patterns:
        if re.search(pattern, query_lower):
            suggestions.append(f"📅 {message}")

    # Check for datetime column references
    datetime_info = get_datetime_columns_info()
    for table_name, columns in datetime_info.items():
        for col_info in columns:
            col_name = col_info['column'].lower()
            if col_name in query_lower or col_name.replace('_', ' ') in query_lower:
                suggestions.append(f"🕒 Will use {col_info['formatted_type']} format for column '{col_info['column']}'")

    return suggestions


# Main entry point
def render_sqlbot(llm):
    """
    Main function to render the SQL Bot interface.
    Called by main.py with llm parameter.
    """
    st.title("🤖 SQL Bot with Enhanced DateTime Support")
    st.write("Ask questions about your data in natural language, and I'll generate SQL queries for you.")

    # Initialize session state
    init_sql_bot_session_state()

    # Show database schema information
    show_schema_info()

    # Use the provided LLM
    if llm is None:
        st.error("❌ No LLM instance available. Please check your configuration.")
        return

    # Get database schema
    schema = get_db_schema()
    schema_text = get_schema_text(schema)

    # Show advanced options
    advanced_options = show_advanced_options()

    # Main query interface
    st.subheader("💬 Ask Your Question")

    # Natural language input
    nl_query = st.text_area(
        "Enter your question about the data:",
        value=st.session_state.get('sql_bot_nl_query_input', ''),
        height=100,
        placeholder="e.g., 'Show me accounts created this year with balance over 10000'"
    )

    # Update session state
    st.session_state['sql_bot_nl_query_input'] = nl_query

    # Buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        generate_sql_button = st.button(
            "🔄 Generate SQL",
            type="primary",
            disabled=not nl_query.strip(),
            use_container_width=True
        )

    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)

    with col3:
        example_button = st.button("💡 Examples", use_container_width=True)

    # Handle button clicks
    if clear_button:
        st.session_state['sql_bot_nl_query_input'] = ""
        st.session_state['sql_bot_generated_query'] = ""
        st.rerun()

    if example_button:
        st.session_state['show_examples'] = not st.session_state.get('show_examples', False)
        st.rerun()

    # Show examples if requested
    if st.session_state.get('show_examples', False):
        show_example_queries()

    # Generate SQL query
    if generate_sql_button and nl_query.strip():
        with st.spinner("🧠 Generating SQL query..."):
            try:
                generated_query = generate_sql_from_nl(
                    nl_query,
                    schema_text,
                    llm,
                    advanced_options['advanced_mode']
                )

                if generated_query:
                    st.session_state['sql_bot_generated_query'] = generated_query
                    st.success("✅ SQL query generated successfully!")
                else:
                    st.error("❌ Failed to generate SQL query.")

            except Exception as e:
                st.error(f"❌ Error generating SQL: {e}")

    # Display and execute generated query
    if st.session_state.get('sql_bot_generated_query'):
        st.subheader("🔍 Generated SQL Query")

        # Allow editing of the generated query
        edited_query = st.text_area(
            "Review and edit the generated SQL query:",
            value=st.session_state['sql_bot_generated_query'],
            height=150,
            key="sql_editor"
        )

        # Query validation
        if advanced_options['validate_queries']:
            is_valid, validation_message = validate_sql_query(edited_query)
            if is_valid:
                st.success(f"✅ {validation_message}")
            else:
                st.error(f"❌ {validation_message}")

        # Execute button
        col1, col2 = st.columns([3, 1])

        with col1:
            execute_button = st.button(
                "▶️ Execute Query",
                type="primary",
                disabled=advanced_options['validate_queries'] and not is_valid,
                use_container_width=True
            )

        with col2:
            save_query_button = st.button("💾 Save Query", use_container_width=True)

        # Execute the query
        if execute_button:
            conn = get_db_connection()
            if conn:
                execute_sql_query(edited_query, conn)
            else:
                st.error("❌ No database connection available.")

        # Save query functionality
        if save_query_button:
            add_query_to_history(edited_query, None)
            st.success("💾 Query saved to history!")

    # Show query history
    with st.expander("📜 Query History"):
        show_query_history()
