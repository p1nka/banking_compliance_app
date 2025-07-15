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
from ai.llm import load_llm
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser


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


def fix_duplicate_columns(df):
    """
    Robust function to fix duplicate column names by adding suffixes.
    """
    if df is None or df.empty:
        return df

    # Create a copy to avoid modifying the original
    df_fixed = df.copy()

    # Get column names
    columns = list(df_fixed.columns)

    # Check for duplicates
    if len(columns) != len(set(columns)):
        st.warning(f"🔧 Detected duplicate columns: {columns}")

        # Create new unique column names
        new_columns = []
        column_counts = {}

        for col in columns:
            if col in column_counts:
                column_counts[col] += 1
                new_col = f"{col}__{column_counts[col]}"  # Use double underscore for clarity
                new_columns.append(new_col)
            else:
                column_counts[col] = 0
                new_columns.append(col)

        # Apply the new column names
        df_fixed.columns = new_columns
        st.success(f"✅ Fixed columns: {columns} → {new_columns}")

        return df_fixed

    return df_fixed


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
    try:
        datetime_info = get_datetime_columns_info()
        if datetime_info:
            schema_text += "\nDATETIME COLUMN DETAILS:\n"
            for table_name, columns in datetime_info.items():
                if columns:
                    schema_text += f"Table {table_name} datetime columns:\n"
                    for col_info in columns:
                        schema_text += f"  - {col_info['column']}: {col_info['formatted_type']} (SQL Server {col_info['type']})\n"
                    schema_text += "\n"
    except:
        pass  # Skip if datetime info not available

    return schema_text


def create_datetime_aware_prompt():
    """
    Enhanced prompt that STRICTLY prevents duplicate columns.
    """

    anti_duplicate_rules = """
🚨 CRITICAL ANTI-DUPLICATE COLUMN RULES:
1. **NEVER select the same column twice in a single SELECT statement**
2. **ALWAYS use aliases for aggregate functions**: COUNT(*) AS account_count, SUM(balance) AS total_balance
3. **NEVER use patterns like**: SELECT column_name, column_name, COUNT(*)
4. **ALWAYS check your SELECT clause for duplicates before generating**

CORRECT Examples:
✅ SELECT account_type, COUNT(*) AS account_count FROM accounts_data GROUP BY account_type
✅ SELECT account_type, SUM(balance) AS total_balance FROM accounts_data GROUP BY account_type
✅ SELECT account_type, AVG(balance) AS average_balance FROM accounts_data GROUP BY account_type

WRONG Examples (NEVER DO THESE):
❌ SELECT account_type, account_type, COUNT(*) FROM accounts_data GROUP BY account_type
❌ SELECT account_type, COUNT(*) FROM accounts_data GROUP BY account_type (missing alias)
❌ SELECT *, account_type FROM accounts_data (duplicate via *)
"""

    aggregation_guidance = """
AGGREGATION & GROUP BY GUIDELINES:
- When a question asks for a "count of", "sum of", "average of", or to "group by", you MUST use a GROUP BY clause.
- **CRITICAL RULE**: When using `GROUP BY`, the `SELECT` statement can ONLY contain the columns listed in the `GROUP BY` clause and aggregate functions (e.g., `COUNT(*)`, `SUM(column)`, `AVG(column)`).
- **DO NOT use `SELECT *` with `GROUP BY`**. This is invalid SQL.
- **ALWAYS use descriptive aliases for aggregate functions** to avoid duplicate column names.

CORRECT Examples:
- Question: "How many accounts per type?"
  SQL: `SELECT account_type, COUNT(*) AS account_count FROM accounts_data GROUP BY account_type`
- Question: "What is the total balance for each account type?"
  SQL: `SELECT account_type, SUM(Current_Balance) AS total_balance FROM accounts_data GROUP BY account_type`
"""

    window_functions_guidance = """
WINDOW & RANKING FUNCTION GUIDELINES:
- Use window functions for questions involving rankings, sequences, or comparisons within a partition.
- **ALWAYS use aliases for window functions**.
- **CRITICAL RULE**: You CANNOT use a window function in a `WHERE` clause directly. You MUST use a Common Table Expression (CTE) or a subquery.

CORRECT Examples with CTEs:
- Question: "Rank accounts by balance"
  SQL: `WITH RankedAccounts AS (SELECT *, RANK() OVER (ORDER BY Current_Balance DESC) as balance_rank FROM accounts_data) SELECT * FROM RankedAccounts WHERE balance_rank <= 10;`
"""

    enhanced_prompt = f"""
You are an expert SQL query generator for SQL Server/Azure SQL Database. Convert the natural language question to a syntactically correct SQL query.

{anti_duplicate_rules}

{aggregation_guidance}

{window_functions_guidance}

Database Schema:
{{schema}}

Question: {{question}}

FINAL CHECK BEFORE RESPONDING:
1. Does your SELECT clause have any duplicate column names? If YES, fix it!
2. Do all aggregate functions have aliases? If NO, add them!
3. Are you following the GROUP BY rules? If NO, fix it!

Generate ONLY the SQL query without any explanation, markdown formatting, or extra text.
"""
    return enhanced_prompt


def enhanced_generate_sql_from_nl(nl_query: str, schema_text: str, llm, is_advanced_mode: bool = False) -> Optional[
    str]:
    """
    SQL BOT
    """
    try:
        enhanced_prompt_template = create_datetime_aware_prompt()

        nl_to_sql_prompt = PromptTemplate.from_template(enhanced_prompt_template)
        nl_to_sql_chain = nl_to_sql_prompt | llm | StrOutputParser()

        sql_query_raw = nl_to_sql_chain.invoke({
            "schema": schema_text,
            "question": nl_query.strip()
        })

        sql_query_generated = clean_sql_query(sql_query_raw)

        if not sql_query_generated or not sql_query_generated.lower().strip().startswith("select"):
            st.error("Generated text does not start with SELECT or is empty.")
            return None

        # VALIDATE: Check for duplicate columns in generated SQL
        duplicate_check_result = validate_sql_for_duplicates(sql_query_generated)
        if not duplicate_check_result['is_valid']:
            st.error(f"🚨 Generated SQL has duplicate columns: {duplicate_check_result['message']}")
            st.code(sql_query_generated, language='sql')
            return None

        return sql_query_generated

    except Exception as e:
        st.error(f"SQL generation error: {e}")
        return get_fallback_response("sql_generation")


def validate_sql_for_duplicates(sql_query: str) -> Dict[str, Any]:
    """
    Validate SQL query for duplicate columns in SELECT clause.
    """
    try:
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return {'is_valid': True, 'message': 'No SELECT clause found'}

        select_clause = select_match.group(1).strip()

        # Handle SELECT *
        if select_clause.strip() == '*':
            return {'is_valid': True, 'message': 'SELECT * is fine'}

        # Split by comma and analyze
        columns = [col.strip() for col in select_clause.split(',')]

        # Extract base column names (remove aliases and functions)
        base_columns = []
        for col in columns:
            # Remove alias (everything after AS)
            if ' AS ' in col.upper():
                base_col = col.split(' AS ')[0].strip()
            else:
                # For functions like COUNT(*), use the whole thing
                base_col = col.strip()

            # Extract actual column name from functions
            if '(' in base_col and ')' in base_col:
                # This is a function, keep as is
                base_columns.append(base_col)
            else:
                # Regular column name
                base_columns.append(base_col)

        # Check for exact duplicates
        duplicates = []
        seen = set()
        for col in base_columns:
            if col in seen:
                duplicates.append(col)
            seen.add(col)

        if duplicates:
            return {
                'is_valid': False,
                'message': f'Duplicate columns found: {duplicates}',
                'columns': base_columns
            }

        return {'is_valid': True, 'message': 'No duplicates found'}

    except Exception as e:
        return {'is_valid': True, 'message': f'Validation error: {e}'}


def identify_datetime_columns_for_query(sql_query: str) -> Optional[List[str]]:
    """
    Try to identify which columns in the query might be datetime columns.
    """
    try:
        datetime_info = get_datetime_columns_info()
        datetime_cols = []
        query_lower = sql_query.lower()
        for table_name, columns in datetime_info.items():
            for col_info in columns:
                col_name = col_info['column']
                if col_name.lower() in query_lower:
                    datetime_cols.append(col_name)
        return datetime_cols if datetime_cols else None
    except:
        return None


def fix_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process DataFrame to fix datetime columns that pandas missed.
    """
    for col in df.columns:
        if df[col].dtype == 'object' and not df[col].empty:
            if is_datetime_column_heuristic(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                except Exception:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m-%d')
                    except Exception:
                        pass
    return df


def is_datetime_column_heuristic(series: pd.Series) -> bool:
    """
    Heuristic to detect if an object column contains datetime values.
    """
    if series.empty:
        return False
    sample = series.dropna().head(10)
    if sample.empty:
        return False
    datetime_count = 0
    for value in sample:
        if isinstance(value, str):
            datetime_patterns = [
                r'\d{4}-\d{2}-\d{2}', r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                r'\d{2}/\d{2}/\d{4}', r'\d{4}/\d{2}/\d{2}', r'\d{2}-\d{2}-\d{4}',
            ]
            for pattern in datetime_patterns:
                if re.search(pattern, str(value)):
                    datetime_count += 1
                    break
    return datetime_count / len(sample) > 0.5


def format_datetime_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format datetime columns for better display in Streamlit.
    """
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if pd.api.types.is_datetime64_any_dtype(formatted_df[col]):
            try:
                has_time = formatted_df[col].dt.time.apply(lambda x: x != pd.Timestamp('00:00:00').time()).any()
                if has_time:
                    formatted_df[col] = formatted_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    formatted_df[col] = formatted_df[col].dt.strftime('%Y-%m-%d')
                formatted_df[col] = formatted_df[col].fillna('')
            except Exception:
                formatted_df[col] = formatted_df[col].astype(str).replace('NaT', '')
        elif pd.api.types.is_timedelta64_dtype(formatted_df[col]):
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
                'Column': col, 'Data Type': dtype, 'Non-Null': non_null,
                'Null': null_count, 'Unique Values': unique_count,
                'Additional Info': extra_info
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True)


def smart_detect_chart_type_debug(results_df: pd.DataFrame):
    """
    Smart chart type detection with extensive debugging and duplicate handling.
    """
    if results_df.empty:
        return None, None, None, None

    # CRITICAL: Ensure no duplicate columns before analysis
    if len(results_df.columns) != len(set(results_df.columns)):
        st.warning("🔧 Fixing duplicate columns in chart detection...")
        results_df = fix_duplicate_columns(results_df)

    columns = list(results_df.columns)

    # Analyze column types
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = results_df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Check for aggregation patterns
    has_count_col = any('count' in col.lower() for col in columns)
    has_sum_col = any(col.lower().startswith(('sum_', 'total_')) for col in columns)
    has_avg_col = any(col.lower().startswith(('avg_', 'average_')) for col in columns)

    # Filter categorical columns
    good_categorical_cols = [
        col for col in categorical_cols
        if not col.lower().endswith(('_id', 'id')) and 1 < results_df[col].nunique() <= 25
    ]

    # Decision logic
    if (has_count_col or has_sum_col or has_avg_col) and good_categorical_cols:
        cat_col = good_categorical_cols[0]

        # Find the numeric column
        if has_count_col:
            num_col = next((col for col in columns if 'count' in col.lower()), None)
        elif has_sum_col:
            num_col = next((col for col in columns if col.lower().startswith(('sum_', 'total_'))), None)
        elif has_avg_col:
            num_col = next((col for col in columns if col.lower().startswith(('avg_', 'average_'))), None)
        else:
            num_col = numeric_cols[0] if numeric_cols else None

        if num_col:
            chart_type = 'pie' if results_df[cat_col].nunique() <= 8 else 'bar'
            title = f"Distribution of {num_col} by {cat_col}"
            return chart_type, cat_col, num_col, title

    # Fallback logic
    if good_categorical_cols:
        cat_col = good_categorical_cols[0]
        chart_type = 'pie' if results_df[cat_col].nunique() <= 8 else 'bar'
        title = f"Distribution of {cat_col}"
        return chart_type, cat_col, None, title

    return None, None, None, None


def create_insights_chart_defensive(data, labels=None, values=None, chart_type='pie', title=None):
    """
    Create a chart with defensive programming against duplicate columns.
    """
    try:
        # CRITICAL: Ensure no duplicate columns before chart creation
        if len(data.columns) != len(set(data.columns)):
            st.warning("🔧 Emergency duplicate fix in chart creation...")
            data = fix_duplicate_columns(data)

        # Validate that required columns exist
        if labels and labels not in data.columns:
            st.error(f"❌ Label column '{labels}' not found in {list(data.columns)}")
            return None

        if values and values not in data.columns:
            st.error(f"❌ Values column '{values}' not found in {list(data.columns)}")
            return None

        # Create the chart based on type
        if chart_type == 'pie' and labels:
            if values and values in data.columns:
                # Use pre-aggregated values
                fig = px.pie(
                    data,
                    names=labels,
                    values=values,
                    title=title
                )
            else:
                # Count occurrences
                counts = data[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                fig = px.pie(
                    counts,
                    names=labels,
                    values='count',
                    title=title
                )

            # Add hover information
            fig.update_traces(
                hovertemplate="<b>%{label}</b><br>" +
                              "Count: %{value:,.0f}<br>" +
                              "Percentage: %{percent}<br>" +
                              "<extra></extra>"
            )
            return fig

        elif chart_type == 'bar':
            if values and values in data.columns:
                # Use pre-aggregated values
                fig = px.bar(
                    data,
                    x=labels,
                    y=values,
                    title=title
                )
            else:
                # Count occurrences
                counts = data[labels].value_counts().reset_index()
                counts.columns = [labels, 'count']
                fig = px.bar(
                    counts,
                    x=labels,
                    y='count',
                    title=title
                )

            # Add hover information and styling
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                              "Count: %{y:,.0f}<br>" +
                              "<extra></extra>"
            )
            fig.update_layout(
                xaxis_title=labels,
                yaxis_title=values if values else 'Count',
                showlegend=False
            )
            return fig

        elif chart_type == 'histogram' and labels:
            fig = px.histogram(data, x=labels, title=title)
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>" +
                              "Count: %{y:,.0f}<br>" +
                              "<extra></extra>"
            )
            return fig

        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            return None

    except Exception as e:
        st.error(f"❌ Chart creation failed: {e}")
        return None


# Add these missing functions to your ui/sqlbot_ui.py file

def show_schema_info():
    """Display database schema information."""
    st.subheader("🗂️ Database Schema")
    try:
        schema = get_db_schema()
        if schema:
            # Show a brief overview
            total_tables = len(schema)
            total_columns = sum(len(columns) for columns in schema.values())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tables", total_tables)
            with col2:
                st.metric("Total Columns", total_columns)

            # Show table names
            st.write("**Available Tables:**")
            for table_name, columns in schema.items():
                st.write(f"- `{table_name}` ({len(columns)} columns)")

            with st.expander("View Detailed Schema", expanded=False):
                st.text(get_schema_text(schema))
        else:
            st.error("Could not retrieve database schema.")
    except Exception as e:
        st.error(f"Schema error: {e}")


def show_advanced_options():
    """Show advanced options for SQL Bot."""
    with st.expander("⚙️ Advanced Options"):
        adv_mode = st.checkbox("Advanced Mode", key='adv_mode_cb')
        val_queries = st.checkbox("Validate Queries", value=True, key='val_q_cb')
        show_debug = st.checkbox("Show Debug Information", key='debug_cb')
        return {
            'advanced_mode': adv_mode,
            'validate_queries': val_queries,
            'show_debug': show_debug
        }


def test_with_manual_sql():
    """
    Test with a manually written SQL query to isolate the issue.
    """
    st.subheader("🔧 Manual SQL Test")

    # Test with a known good SQL query
    good_sql = "SELECT account_type, COUNT(*) AS account_count FROM accounts_data GROUP BY account_type"

    st.write("**Testing with manually written SQL:**")
    st.code(good_sql, language='sql')

    if st.button("Execute Manual SQL"):
        conn = get_db_connection()
        if conn:
            try:
                results_df = pd.read_sql(good_sql, conn)
                st.write("**Results from manual SQL:**")
                st.write(f"Columns: {list(results_df.columns)}")
                st.dataframe(results_df)

                # Test visualization with known good data
                if 'auto_generate_visualizations_enhanced' in globals():
                    auto_generate_visualizations_enhanced(results_df)
                else:
                    st.info("Visualization function not available")

            except Exception as e:
                st.error(f"Manual SQL failed: {e}")
                st.exception(e)


def validate_sql_query(sql_query: str) -> Tuple[bool, str]:
    """Enhanced SQL query validation with duplicate checking."""
    if not sql_query or not sql_query.strip():
        return False, "Query is empty"

    cleaned_query = re.sub(r'--.*?\n', '', sql_query).strip()
    query_lower = cleaned_query.lower()

    if not query_lower.startswith('select'):
        return False, "Query must start with SELECT"

    # Check for duplicate columns if function exists
    try:
        if 'validate_sql_for_duplicates' in globals():
            duplicate_check = validate_sql_for_duplicates(sql_query)
            if not duplicate_check['is_valid']:
                return False, f"Duplicate columns detected: {duplicate_check['message']}"
    except:
        pass  # Skip duplicate check if function not available

    # Check for GROUP BY with SELECT *
    if 'group by' in query_lower:
        select_part_match = re.search(r'select(.*?)from', query_lower, re.DOTALL)
        if select_part_match:
            select_part = select_part_match.group(1).strip()
            if select_part == '*':
                return False, "Invalid query: Cannot use 'SELECT *' with 'GROUP BY'."

    # Check for dangerous keywords
    dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', query_lower):
            return False, f"Dangerous keyword '{keyword.upper()}' not allowed."

    # Check balanced parentheses
    if cleaned_query.count('(') != cleaned_query.count(')'):
        return False, "Unbalanced parentheses"

    return True, "Query validation passed"


def show_query_history():
    """Display query history."""
    try:
        conn = get_db_connection()
        if conn:
            try:
                df_history = pd.read_sql("SELECT TOP 10 query_text FROM sql_query_history ORDER BY created_at DESC",
                                         conn)
                if not df_history.empty:
                    st.write("**📜 Recent Queries:**")
                    for idx, row in df_history.iterrows():
                        if st.button(f"📋 Use: {row['query_text'][:50]}...", key=f"hist_btn_{idx}"):
                            st.session_state['sql_bot_generated_query'] = row['query_text']
                            st.rerun()
                else:
                    st.info("No query history found.")
            except Exception as e:
                st.info(f"Query history table not available: {e}")
    except Exception as e:
        st.write(f"Query history unavailable: {e}")


def validate_sql_for_duplicates(sql_query: str) -> Dict[str, Any]:
    """
    Validate SQL query for duplicate columns in SELECT clause.
    """
    try:
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
        if not select_match:
            return {'is_valid': True, 'message': 'No SELECT clause found'}

        select_clause = select_match.group(1).strip()

        # Handle SELECT *
        if select_clause.strip() == '*':
            return {'is_valid': True, 'message': 'SELECT * is fine'}

        # Split by comma and analyze
        columns = [col.strip() for col in select_clause.split(',')]

        # Extract base column names (remove aliases and functions)
        base_columns = []
        for col in columns:
            # Remove alias (everything after AS)
            if ' AS ' in col.upper():
                base_col = col.split(' AS ')[0].strip()
            else:
                # For functions like COUNT(*), use the whole thing
                base_col = col.strip()

            # Extract actual column name from functions
            if '(' in base_col and ')' in base_col:
                # This is a function, keep as is
                base_columns.append(base_col)
            else:
                # Regular column name
                base_columns.append(base_col)

        # Check for exact duplicates
        duplicates = []
        seen = set()
        for col in base_columns:
            if col in seen:
                duplicates.append(col)
            seen.add(col)

        if duplicates:
            return {
                'is_valid': False,
                'message': f'Duplicate columns found: {duplicates}',
                'columns': base_columns
            }

        return {'is_valid': True, 'message': 'No duplicates found'}

    except Exception as e:
        return {'is_valid': True, 'message': f'Validation error: {e}'}


def show_example_queries():
    """Show example queries that users can try."""
    st.subheader("💡 Example Queries")

    examples = [
        ("Basic Query", "Show me all accounts", "basic"),
        ("Filter Query", "Show accounts with balance over 10000", "filter"),
        ("Count Query", "How many accounts do we have?", "count"),
        ("Group Query", "Show account count by type", "group"),
        ("Ranking Query", "Show the top 3 accounts per type by balance", "ranking")
    ]

    st.write("**Click any example to try it:**")
    cols = st.columns(len(examples))
    for i, (title, query, key) in enumerate(examples):
        with cols[i]:
            if st.button(title, key=f"example_{key}", use_container_width=True, help=f"Try: {query}"):
                st.session_state['sql_bot_nl_query_input'] = query
                st.rerun()

    # Show datetime examples if function exists
    try:
        show_datetime_examples()
    except:
        pass


def show_datetime_examples():
    """Show specific examples for datetime queries."""
    st.subheader("🕒 DateTime Query Examples")
    examples = [
        ("Recent Activity", "Show accounts with activity in the last 30 days", "recent"),
        ("This Year", "Show all accounts created this year", "this_year"),
        ("Monthly Trends", "Show count of accounts created by month", "monthly")
    ]
    cols = st.columns(len(examples))
    for i, (title, query, key) in enumerate(examples):
        if cols[i].button(title, key=f"dt_ex_{key}", use_container_width=True, help=f"Try: {query}"):
            st.session_state['sql_bot_nl_query_input'] = query
            st.rerun()


def auto_generate_visualizations_enhanced(results_df: pd.DataFrame):
    """
    Simple visualization function - placeholder until full implementation.
    """
    if results_df is None or results_df.empty:
        st.info("No data to visualize.")
        return

    st.subheader("📈 Auto-Generated Visualizations")

    try:
        # Simple bar chart for categorical data
        if len(results_df.columns) >= 2:
            # Assume first column is categorical, second is numeric
            cat_col = results_df.columns[0]
            num_col = results_df.columns[1]

            # Create a simple bar chart
            fig = px.bar(results_df, x=cat_col, y=num_col, title=f"{num_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

            # Show details
            with st.expander("📊 Visualization Details"):
                st.write(f"**Chart Type:** Bar Chart")
                st.write(f"**X-axis:** {cat_col}")
                st.write(f"**Y-axis:** {num_col}")
                st.write(f"**Data:** {len(results_df)} rows")
        else:
            st.info("Need at least 2 columns for visualization")

    except Exception as e:
        st.error(f"Visualization error: {e}")
        st.write("**Available columns:**", list(results_df.columns))
        st.dataframe(results_df.head())


def enhanced_execute_sql_query(sql_query: str, conn):
    """
    Simple SQL execution function.
    """
    try:
        cleaned_query = clean_sql_query(sql_query)
        st.info("🔍 **Executing SQL Query**")
        st.code(cleaned_query, language='sql')

        with st.spinner("⏳ Executing query..."):
            try:
                # Execute the SQL query
                results_df = pd.read_sql(cleaned_query, conn)

                # Check for duplicates and fix if needed
                original_columns = list(results_df.columns)
                has_duplicates = len(original_columns) != len(set(original_columns))

                if has_duplicates:
                    st.error(f"🚨 SQL Query returned duplicate columns: {original_columns}")
                    # Simple duplicate fix
                    new_columns = []
                    counts = {}
                    for col in original_columns:
                        if col in counts:
                            counts[col] += 1
                            new_columns.append(f"{col}_{counts[col]}")
                        else:
                            counts[col] = 0
                            new_columns.append(col)
                    results_df.columns = new_columns
                    st.success(f"✅ Fixed columns: {new_columns}")
                else:
                    st.success("✅ No duplicate columns detected")

            except Exception as sql_error:
                st.error(f"SQL execution failed: {sql_error}")
                return

        st.success(f"✅ Query executed successfully! Returned {len(results_df)} rows.")

        if not results_df.empty:
            # Show debug information
            with st.expander("🔍 SQL Execution Debug Info", expanded=False):
                st.write("**DataFrame Info:**")
                st.write(f"- Shape: {results_df.shape}")
                st.write(f"- Columns: {list(results_df.columns)}")
                st.write("**Sample Data:**")
                st.dataframe(results_df.head(5))

            # Generate visualizations
            auto_generate_visualizations_enhanced(results_df)

            # Show the data table
            st.subheader("📋 Data Table")
            st.dataframe(results_df, use_container_width=True, height=400)

            # Add to query history
            add_query_to_history(cleaned_query, results_df)
        else:
            st.info("✅ Query executed successfully but returned no results.")

    except Exception as e:
        st.error(f"❌ Query execution error: {e}")
        st.exception(e)


# Add these helper functions as well

def clean_sql_query(sql_query_raw: str) -> str:
    """Clean SQL query by removing markdown formatting."""
    sql_query = re.sub(r'```sql\n?|```\n?', '', sql_query_raw)
    return sql_query.strip().rstrip(';')


def add_query_to_history(sql_query: str, results_df: Optional[pd.DataFrame], error: Optional[str] = None):
    """Add query to session state history."""
    history = st.session_state.get('sql_bot_queries_history', [])
    history.append({
        'timestamp': datetime.now(),
        'sql_query': sql_query,
        'row_count': len(results_df) if results_df is not None else 0,
        'success': error is None,
        'error': error
    })
    st.session_state['sql_bot_queries_history'] = history[-50:]


def generate_sql_from_nl(nl_query: str, schema_text: str, llm, is_advanced_mode: bool = False) -> Optional[str]:
    """Simple SQL generation function."""
    try:
        if 'enhanced_generate_sql_from_nl' in globals():
            return enhanced_generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode)
        else:
            # Fallback: return a simple query
            st.warning("Using fallback SQL generation")
            return f"SELECT account_type, COUNT(*) AS account_count FROM accounts_data GROUP BY account_type -- Generated for: {nl_query}"
    except Exception as e:
        st.error(f"SQL generation error: {e}")
        return None


def execute_sql_query(sql_query: str, conn):
    """Execute SQL query wrapper."""
    return enhanced_execute_sql_query(sql_query, conn)


def get_fallback_response(error_type: str) -> str:
    """Get fallback SQL query."""
    return "SELECT account_type, COUNT(*) AS account_count FROM accounts_data GROUP BY account_type"


def auto_generate_visualizations_enhanced(results_df: pd.DataFrame):
    """
    Enhanced auto-visualization with robust duplicate column handling.
    """
    if results_df is None or results_df.empty:
        st.info("No data to visualize.")
        return

    st.subheader("📈 Auto-Generated Visualizations")

    try:
        # STEP 1: Create a clean copy and fix any column issues
        clean_df = results_df.copy()

        # STEP 2: Debug - show what we received
        with st.expander("🔍 Debug: Data Structure", expanded=False):
            st.write("**Original DataFrame Info:**")
            st.write(f"- Shape: {clean_df.shape}")
            st.write(f"- Columns: {list(clean_df.columns)}")
            st.write(f"- Column types: {dict(clean_df.dtypes)}")
            st.write("**Sample data:**")
            st.dataframe(clean_df.head(3))

        # STEP 3: Fix duplicate columns if they exist
        original_columns = list(clean_df.columns)
        if len(original_columns) != len(set(original_columns)):
            st.warning(f"🔧 Detected duplicate columns: {original_columns}")

            # Create unique column names
            new_columns = []
            seen_columns = {}

            for col in original_columns:
                if col in seen_columns:
                    seen_columns[col] += 1
                    new_col = f"{col}_v{seen_columns[col]}"
                    new_columns.append(new_col)
                    st.write(f"Renamed duplicate: {col} → {new_col}")
                else:
                    seen_columns[col] = 0
                    new_columns.append(col)

            clean_df.columns = new_columns
            st.success(f"✅ Fixed duplicate columns: {new_columns}")

        # STEP 4: Identify the data structure
        columns = list(clean_df.columns)
        st.write(f"**Working with columns:** {columns}")

        # Look for the pattern: Account_Type + numeric columns
        account_type_col = None
        numeric_value_cols = []

        # Find Account_Type column (original or renamed)
        for col in columns:
            if 'account_type' in col.lower():
                account_type_col = col
                break

        # Find numeric columns
        numeric_cols = clean_df.select_dtypes(include=['number']).columns.tolist()

        # Filter out index-like columns and find value columns
        for col in numeric_cols:
            if not col.lower().endswith('_id') and clean_df[col].sum() > 0:
                numeric_value_cols.append(col)

        st.write(f"**Detected Account Type column:** {account_type_col}")
        st.write(f"**Detected numeric columns:** {numeric_value_cols}")

        # STEP 5: Create visualization based on detected pattern
        if account_type_col and numeric_value_cols:
            # This looks like account type analysis data
            value_col = numeric_value_cols[0]  # Use first numeric column

            st.write(f"**Creating bar chart: {value_col} by {account_type_col}**")

            try:
                # Ensure no duplicate columns in the specific subset we're using
                viz_data = clean_df[[account_type_col, value_col]].copy()

                # Double-check for duplicates in our subset
                if len(viz_data.columns) != len(set(viz_data.columns)):
                    st.error("Still have duplicates in visualization data!")
                    st.write(f"Viz data columns: {list(viz_data.columns)}")
                    return

                # Create the bar chart
                fig = px.bar(
                    viz_data,
                    x=account_type_col,
                    y=value_col,
                    title=f"{value_col.replace('_', ' ').title()} by Account Type",
                    labels={
                        account_type_col: "Account Type",
                        value_col: value_col.replace('_', ' ').title()
                    }
                )

                # Customize the chart
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>" +
                                  f"{value_col}: %{{y:,.0f}}<br>" +
                                  "<extra></extra>"
                )

                fig.update_layout(
                    xaxis_title="Account Type",
                    yaxis_title=value_col.replace('_', ' ').title(),
                    showlegend=False,
                    height=500
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                st.success("✅ Bar chart created successfully!")

                # Show visualization details
                with st.expander("📊 Visualization Details"):
                    st.write(f"**Chart Type:** Bar Chart")
                    st.write(f"**X-axis:** {account_type_col} (Account Type)")
                    st.write(f"**Y-axis:** {value_col}")
                    st.write(f"**Data Points:** {len(clean_df)} account types")

                    # Show the data used for visualization
                    st.write("**Data used for chart:**")
                    st.dataframe(viz_data)

                    # Show summary statistics
                    if pd.api.types.is_numeric_dtype(clean_df[value_col]):
                        st.write("**Summary Statistics:**")
                        stats_col1, stats_col2, stats_col3 = st.columns(3)
                        with stats_col1:
                            st.metric("Total", f"{clean_df[value_col].sum():,.0f}")
                        with stats_col2:
                            st.metric("Average", f"{clean_df[value_col].mean():.1f}")
                        with stats_col3:
                            st.metric("Max", f"{clean_df[value_col].max():,.0f}")

                # Try to create additional charts if we have more numeric columns
                if len(numeric_value_cols) > 1:
                    st.write("**Additional Visualizations:**")

                    # Create a comparison chart with multiple metrics
                    try:
                        comparison_data = clean_df[[account_type_col] + numeric_value_cols[:3]].copy()  # Max 3 metrics

                        # Melt the data for grouped bar chart
                        melted_data = comparison_data.melt(
                            id_vars=[account_type_col],
                            var_name='Metric',
                            value_name='Value'
                        )

                        fig2 = px.bar(
                            melted_data,
                            x=account_type_col,
                            y='Value',
                            color='Metric',
                            title="Comparison of Multiple Metrics by Account Type",
                            barmode='group'
                        )

                        fig2.update_layout(
                            xaxis_title="Account Type",
                            yaxis_title="Value",
                            height=500
                        )

                        st.plotly_chart(fig2, use_container_width=True)
                        st.success("✅ Comparison chart created!")

                    except Exception as comp_error:
                        st.info(f"Could not create comparison chart: {comp_error}")

            except Exception as bar_error:
                st.error(f"❌ Bar chart creation failed: {bar_error}")
                st.write("**Error details:**")
                st.exception(bar_error)

                # Fallback: show data table
                st.write("**Fallback: Data Display**")
                st.dataframe(clean_df)

        else:
            st.info("Could not identify appropriate columns for account type visualization.")
            st.write("**Available data:**")
            st.dataframe(clean_df)

            # Try a simple chart with any available data
            if len(numeric_value_cols) > 0:
                try:
                    # Simple histogram of the first numeric column
                    first_numeric = numeric_value_cols[0]
                    fig = px.histogram(clean_df, x=first_numeric, title=f"Distribution of {first_numeric}")
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(f"Created histogram for {first_numeric}")
                except Exception as hist_error:
                    st.error(f"Histogram creation failed: {hist_error}")

    except Exception as e:
        st.error(f"❌ Complete visualization failure: {e}")
        with st.expander("Show full error details"):
            st.exception(e)
            st.write("**DataFrame info:**")
            try:
                st.write(f"- Shape: {results_df.shape}")
                st.write(f"- Columns: {list(results_df.columns)}")
                st.write("**Raw data:**")
                st.dataframe(results_df.head())
            except:
                st.error("Could not even display basic DataFrame info")


def enhanced_execute_sql_query(sql_query: str, conn):
    """
    Enhanced SQL query execution with IMMEDIATE duplicate column fixing.
    """
    try:
        cleaned_query = clean_sql_query(sql_query)
        st.info("🔍 **Executing SQL Query**")
        st.code(cleaned_query, language='sql')

        with st.spinner("⏳ Executing query..."):
            try:
                # Execute the SQL query
                results_df = pd.read_sql(cleaned_query, conn)

                # IMMEDIATE DUPLICATE CHECK AND FIX
                original_columns = list(results_df.columns)
                has_duplicates = len(original_columns) != len(set(original_columns))

                if has_duplicates:
                    st.error(f"🚨 SQL Query returned duplicate columns: {original_columns}")
                    results_df = fix_duplicate_columns(results_df)
                    st.write("**Columns after fixing:**", list(results_df.columns))
                else:
                    st.success("✅ No duplicate columns detected")

            except Exception as sql_error:
                st.error(f"SQL execution failed: {sql_error}")
                return

            # Fix any datetime columns
            results_df = fix_datetime_columns(results_df)

        st.success(f"✅ Query executed successfully! Returned {len(results_df)} rows.")

        if not results_df.empty:
            # Show debug information if enabled
            with st.expander("🔍 SQL Execution Debug Info", expanded=False):
                st.write("**DataFrame Info:**")
                st.write(f"- Shape: {results_df.shape}")
                st.write(f"- Columns: {list(results_df.columns)}")
                st.write("**Sample Data:**")
                st.dataframe(results_df.head(5))

            # Generate visualizations with the FIXED DataFrame
            auto_generate_visualizations_enhanced(results_df)

            # Show the data table
            st.subheader("📋 Data Table")
            display_df = format_datetime_for_display(results_df)
            st.dataframe(display_df, use_container_width=True, height=400)

            # Show column information
            show_enhanced_column_info(results_df)

            # Add to query history
            add_query_to_history(cleaned_query, results_df)
        else:
            st.info("✅ Query executed successfully but returned no results.")

    except Exception as e:
        st.error(f"❌ Query execution error: {e}")
        st.exception(e)


def show_datetime_examples():
    """Show specific examples for datetime queries."""
    st.subheader("🕒 DateTime Query Examples")
    examples = [
        ("Recent Activity", "Show accounts with activity in the last 30 days", "recent"),
        ("This Year", "Show all accounts created this year", "this_year"),
        ("Monthly Trends", "Show count of accounts created by month", "monthly")
    ]
    cols = st.columns(len(examples))
    for i, (title, query, key) in enumerate(examples):
        if cols[i].button(title, key=f"dt_ex_{key}", use_container_width=True, help=f"Try: {query}"):
            st.session_state['sql_bot_nl_query_input'] = query
            st.rerun()


def generate_sql_from_nl(nl_query: str, schema_text: str, llm, is_advanced_mode: bool = False) -> Optional[str]:
    return enhanced_generate_sql_from_nl(nl_query, schema_text, llm, is_advanced_mode)


def execute_sql_query(sql_query: str, conn):
    return enhanced_execute_sql_query(sql_query, conn)


def clean_sql_query(sql_query_raw: str) -> str:
    sql_query = re.sub(r'```sql\n?|```\n?', '', sql_query_raw)
    return sql_query.strip().rstrip(';')


def get_fallback_response(error_type: str) -> str:
    return {
        "sql_generation": "SELECT account_type, COUNT(*) AS account_count FROM accounts_data GROUP BY account_type"}.get(
        error_type, "SELECT 1 as status")


def add_query_to_history(sql_query: str, results_df: Optional[pd.DataFrame], error: Optional[str] = None):
    history = st.session_state.get('sql_bot_queries_history', [])
    history.append({
        'timestamp': datetime.now(),
        'sql_query': sql_query,
        'row_count': len(results_df) if results_df is not None else 0,
        'success': error is None,
        'error': error
    })
    st.session_state['sql_bot_queries_history'] = history[-50:]


def show_example_queries():
    """Show example queries that users can try."""
    st.subheader("💡 Example Queries")

    examples = [
        ("Basic Query", "Show me all accounts", "basic"),
        ("Filter Query", "Show accounts with balance over 10000", "filter"),
        ("Count Query", "How many accounts do we have?", "count"),  # ← Make sure this line is exactly like this
        ("Group Query", "Show account count by type", "group"),
        ("Ranking Query", "Show the top 3 accounts per type by balance", "ranking")
    ]


def render_sqlbot(llm):
    """Main function to render the SQL Bot interface."""
    st.title("🤖 Advanced SQL Bot with Duplicate Column Protection")
    st.write(
        "Ask questions about your data in natural language. The bot will generate and execute SQL queries with intelligent visualizations.")

    # Initialize session state
    init_sql_bot_session_state()

    # Show database schema
    show_schema_info()

    # Check if LLM is available
    if llm is None:
        st.error("❌ No LLM instance available. Please check your API configuration.")
        return

    # Get schema for LLM
    schema_text = get_schema_text(get_db_schema())
    if not schema_text or "No schema available" in schema_text:
        st.error("❌ No database schema available. Please check your database connection.")
        return

    # Show advanced options
    advanced_options = show_advanced_options()

    # Manual SQL test section (only in debug mode)
    if advanced_options.get('show_debug'):
        test_with_manual_sql()
        st.markdown("---")

    # Main query input
    st.subheader("🔤 Natural Language Query")
    nl_query = st.text_area(
        "Enter your question:",
        key='sql_bot_nl_query_input',
        placeholder="e.g., 'Show account count by type' or 'How many accounts do we have by type?'",
        height=100
    )

    # Generate SQL button
    col1, col2 = st.columns([1, 1])
    with col1:
        generate_button = st.button(
            "🔄 Generate SQL",
            type="primary",
            disabled=not nl_query.strip(),
            use_container_width=True
        )

    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear_button:
        st.session_state['sql_bot_nl_query_input'] = ""
        st.session_state['sql_bot_generated_query'] = ""
        st.rerun()

    # Generate SQL when button is clicked
    if generate_button and nl_query.strip():
        with st.spinner("🧠 Generating SQL query..."):
            generated_query = generate_sql_from_nl(
                nl_query,
                schema_text,
                llm,
                advanced_options['advanced_mode']
            )

            if generated_query:
                st.session_state['sql_bot_generated_query'] = generated_query
                st.success("✅ SQL query generated successfully!")

                if advanced_options.get('show_debug'):
                    with st.expander("🔍 Debug Information"):
                        st.write("**Generated Query:**")
                        st.code(generated_query, language='sql')

                        # Validate for duplicates
                        try:
                            duplicate_check = validate_sql_for_duplicates(generated_query)
                            if duplicate_check['is_valid']:
                                st.success("✅ No duplicate columns detected in generated SQL")
                            else:
                                st.error(f"🚨 Duplicate columns detected: {duplicate_check['message']}")
                        except:
                            st.info("Could not validate for duplicates")

                        st.write("**Query Length:**", len(generated_query))
                        st.write("**Schema Used:**", len(schema_text), "characters")
            else:
                st.error("❌ Failed to generate SQL query. Please try rephrasing your question.")

    # Show and edit generated query
    if st.session_state.get('sql_bot_generated_query'):
        st.subheader("📝 Generated SQL Query")

        edited_query = st.text_area(
            "Review and edit the generated query:",
            key="sql_editor",
            value=st.session_state['sql_bot_generated_query'],
            height=150
        )

        # Validate query if validation is enabled
        if advanced_options.get('validate_queries', True):
            is_valid, validation_message = validate_sql_query(edited_query)
            if is_valid:
                st.success(f"✅ {validation_message}")
            else:
                st.error(f"❌ {validation_message}")
        else:
            is_valid = True

        # Execute query button
        execute_button = st.button(
            "▶️ Execute Query",
            type="primary",
            disabled=not is_valid,
            use_container_width=True
        )

        if execute_button and is_valid:
            conn = get_db_connection()
            if conn:
                execute_sql_query(edited_query, conn)
            else:
                st.error("❌ No database connection available. Please check your database configuration.")

    # Show examples and history in an expandable section
    with st.expander("💡 Examples & History", expanded=False):
        tab1, tab2 = st.tabs(["📚 Examples", "📜 History"])

        with tab1:
            show_example_queries()

        with tab2:
            show_query_history()

            # Show session history
            if st.session_state.get('sql_bot_queries_history'):
                st.write("**📋 Session History:**")
                history = st.session_state['sql_bot_queries_history'][-10:]  # Last 10 queries

                for i, query_info in enumerate(reversed(history)):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])

                        with col1:
                            query_preview = query_info['sql_query'][:60] + "..." if len(
                                query_info['sql_query']) > 60 else query_info['sql_query']
                            if st.button(f"📋 {query_preview}", key=f"session_hist_{i}"):
                                st.session_state['sql_bot_generated_query'] = query_info['sql_query']
                                st.rerun()

                        with col2:
                            if query_info['success']:
                                st.success(f"✅ {query_info['row_count']} rows")
                            else:
                                st.error("❌ Failed")

                        with col3:
                            st.caption(query_info['timestamp'].strftime("%H:%M:%S"))

    # Footer with tips
    st.markdown("---")
    st.markdown("""
    **💡 Tips for better results:**
    - Be specific about what you want to see
    - Use proper column names when you know them
    - Try questions like "Show me...", "How many...", "What is the..."
    - For aggregations, use "count by", "sum of", "average of"
    - The system automatically prevents duplicate column errors
    - Use the debug mode to see exactly what SQL is generated
    """)

    # Debug footer
    if advanced_options.get('show_debug'):
        st.markdown("---")
        st.markdown("**🔧 Debug Mode Active** - Additional validation and error checking enabled")

