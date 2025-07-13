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
from ai.visualizations import generate_plot, create_insights_chart


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
    Create an enhanced SQL generation prompt that's aware of window functions, aggregations, and datetime column types.
    """
    aggregation_guidance = """
AGGREGATION & GROUP BY GUIDELINES:
- When a question asks for a "count of", "sum of", "average of", or to "group by", you MUST use a GROUP BY clause.
- **CRITICAL RULE**: When using `GROUP BY`, the `SELECT` statement can ONLY contain the columns listed in the `GROUP BY` clause and aggregate functions (e.g., `COUNT(*)`, `SUM(column)`, `AVG(column)`).
- **DO NOT use `SELECT *` with `GROUP BY`**. This is invalid SQL.

CORRECT Examples:
- Question: "How many accounts per type?"
  SQL: `SELECT account_type, COUNT(*) AS number_of_accounts FROM accounts_data GROUP BY account_type`
- Question: "What is the total balance for each account type?"
  SQL: `SELECT account_type, SUM(Current_Balance) AS total_balance FROM accounts_data GROUP BY account_type`
"""

    # FIX: Added a new, dedicated section to teach the AI about window functions.
    window_functions_guidance = """
WINDOW & RANKING FUNCTION GUIDELINES:
- Use window functions for questions involving rankings, sequences, or comparisons within a partition.
- Keywords like "rank", "top N per group", "most recent for each", "compare to average in group" indicate a window function is needed.
- **`ROW_NUMBER()`**: Use for sequential numbering.
- **`RANK()` / `DENSE_RANK()`**: Use for ranking with ties.
- **`PARTITION BY`**: Use to divide rows into groups (e.g., `PARTITION BY account_type`).
- **CRITICAL RULE**: You CANNOT use a window function in a `WHERE` clause directly. You MUST use a Common Table Expression (CTE) or a subquery.

CORRECT Examples with CTEs:
- Question: "Rank accounts by balance"
  SQL: `WITH RankedAccounts AS (SELECT *, RANK() OVER (ORDER BY Current_Balance DESC) as balance_rank FROM accounts_data) SELECT * FROM RankedAccounts WHERE balance_rank <= 10;`
- Question: "Find the top 3 highest balance accounts for each account_type"
  SQL: `WITH RankedAccounts AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY account_type ORDER BY Current_Balance DESC) as rn FROM accounts_data) SELECT * FROM RankedAccounts WHERE rn <= 3;`
- Question: "Show each account's balance and the average balance for its type"
  SQL: `SELECT account_id, account_type, Current_Balance, AVG(Current_Balance) OVER (PARTITION BY account_type) as avg_balance_for_type FROM accounts_data;`
"""

    datetime_guidance = """
DATETIME GUIDELINES:
- Use SQL Server date functions like `GETDATE()`, `DATEADD()`, `YEAR()`, `MONTH()`.
"""

    enhanced_prompt = f"""
You are an expert SQL query generator for SQL Server/Azure SQL Database. Convert the natural language question to a syntactically correct SQL query.

{aggregation_guidance}
{window_functions_guidance}
{datetime_guidance}

Database Schema:
{{schema}}

Question: {{question}}

CRITICAL INSTRUCTIONS:
1. Adhere strictly to the AGGREGATION, WINDOW/RANKING, and DATETIME guidelines.
2. For ranking or 'top N per group' queries, PREFER using a Common Table Expression (CTE) as shown in the examples.
3. Use the EXACT column names and types shown in the schema.
4. Generate ONLY the SQL query without any explanation, markdown formatting, or extra text.
"""
    return enhanced_prompt


def enhanced_generate_sql_from_nl(nl_query: str, schema_text: str, llm, is_advanced_mode: bool = False) -> Optional[
    str]:
    """
    Enhanced SQL generation with awareness of advanced SQL features.
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
    query_lower = sql_query.lower()
    for table_name, columns in datetime_info.items():
        for col_info in columns:
            col_name = col_info['column']
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


def enhanced_execute_sql_query(sql_query: str, conn):
    """
    Enhanced SQL query execution with better datetime handling and visualization.py integration.
    """
    try:
        cleaned_query = clean_sql_query(sql_query)
        st.info("🔍 **Executing SQL Query**")
        st.code(cleaned_query, language='sql')
        with st.spinner("⏳ Executing query..."):
            try:
                datetime_columns = identify_datetime_columns_for_query(cleaned_query)
                if datetime_columns:
                    st.info(f"🕒 Detected datetime columns: {', '.join(datetime_columns)}")
                    results_df = pd.read_sql(cleaned_query, conn, parse_dates=datetime_columns)
                else:
                    results_df = pd.read_sql(cleaned_query, conn)
            except Exception as parse_error:
                st.warning(f"Advanced datetime parsing failed, using basic method: {parse_error}")
                results_df = pd.read_sql(cleaned_query, conn)
            results_df = fix_datetime_columns(results_df)

        st.subheader("📊 Query Results")
        if not results_df.empty:
            auto_generate_visualizations_enhanced(results_df)
            st.subheader("📋 Data Table")
            st.dataframe(format_datetime_for_display(results_df), use_container_width=True, height=400)
            show_enhanced_column_info(results_df)
            add_query_to_history(cleaned_query, results_df)
        else:
            st.info("✅ Query executed successfully but returned no results.")
            add_query_to_history(cleaned_query, results_df)
    except Exception as e:
        st.error(f"❌ Query execution error: {e}")
        add_query_to_history(sql_query, None, error=str(e))


def auto_generate_visualizations_enhanced(results_df: pd.DataFrame):
    """
    Enhanced auto-visualization using functions from visualization.py
    """
    if results_df is None or results_df.empty: return
    st.subheader("📈 Auto-Generated Visualizations")
    numeric_cols = results_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = [col for col in results_df.select_dtypes(include=['object', 'category']).columns.tolist()
                        if not col.lower().endswith('_id') and 1 < results_df[col].nunique() <= 20]
    date_cols = [col for col in results_df.columns if pd.api.types.is_datetime64_any_dtype(results_df[col])]

    # Simplified plot generation
    if categorical_cols: st.plotly_chart(create_insights_chart(results_df, categorical_cols[0], 'bar'),
                                         use_container_width=True)
    if numeric_cols: st.plotly_chart(create_insights_chart(results_df, numeric_cols[0], 'histogram'),
                                     use_container_width=True)
    if date_cols and numeric_cols:
        plot_spec = {"plot_type": "scatter", "x_column": date_cols[0], "y_column": numeric_cols[0]}
        chart, _ = generate_plot(plot_spec, results_df)
        if chart:
            chart.update_traces(mode='lines+markers')
            st.plotly_chart(chart, use_container_width=True)


def show_datetime_examples():
    """
    Show specific examples for datetime queries with click-to-try functionality.
    """
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
    return {"sql_generation": "SELECT TOP 10 * FROM accounts_data"}.get(error_type, "SELECT 1 as status")


def add_query_to_history(sql_query: str, results_df: Optional[pd.DataFrame], error: Optional[str] = None):
    history = st.session_state.get('sql_bot_queries_history', [])
    history.append({
        'timestamp': datetime.now(), 'sql_query': sql_query,
        'row_count': len(results_df) if results_df is not None else 0,
        'success': error is None, 'error': error
    })
    st.session_state['sql_bot_queries_history'] = history[-50:]


def show_example_queries():
    """Show example queries that users can try."""
    st.subheader("💡 Example Queries")

    examples = [
        ("Basic Query", "Show me all accounts", "basic"),
        ("Filter Query", "Show accounts with balance over 10000", "filter"),
        ("Count Query", "How many accounts do we have?", "count"),
        ("Group Query", "Show account count by type", "group"),
        # FIX: Added a new example for the ranking capability.
        ("Ranking Query", "Show the top 3 accounts per type by balance", "ranking")
    ]

    st.write("**Click any example to try it:**")
    cols = st.columns(len(examples))
    for i, (title, query, key) in enumerate(examples):
        with cols[i]:
            if st.button(title, key=f"example_{key}", use_container_width=True, help=f"Try: {query}"):
                st.session_state['sql_bot_nl_query_input'] = query
                st.rerun()
    show_datetime_examples()


def show_query_history():
    """Display query history."""
    try:
        conn = get_db_connection()
        if conn:
            df_history = pd.read_sql("SELECT TOP 10 query_text FROM sql_query_history ORDER BY created_at DESC", conn)
            if not df_history.empty:
                st.write("**📜 Recent Queries:**")
                for idx, row in df_history.iterrows():
                    if st.button(f"📋 Use: {row['query_text'][:50]}...", key=f"hist_btn_{idx}"):
                        st.session_state['sql_bot_generated_query'] = row['query_text']
                        st.rerun()
    except Exception as e:
        st.write(f"Query history unavailable: {e}")


def validate_sql_query(sql_query: str) -> Tuple[bool, str]:
    """Enhanced SQL query validation."""
    if not sql_query or not sql_query.strip(): return False, "Query is empty"
    cleaned_query = re.sub(r'--.*?\n', '', sql_query).strip()
    query_lower = cleaned_query.lower()

    if not query_lower.startswith('select'): return False, "Query must start with SELECT"

    if 'group by' in query_lower:
        select_part_match = re.search(r'select(.*?)from', query_lower, re.DOTALL)
        if select_part_match:
            select_part = select_part_match.group(1).strip()
            if select_part == '*':
                return False, "Invalid query: Cannot use 'SELECT *' with 'GROUP BY'."

    dangerous_keywords = ['drop', 'delete', 'truncate', 'alter', 'create', 'insert', 'update']
    for keyword in dangerous_keywords:
        if re.search(r'\b' + keyword + r'\b', query_lower):
            return False, f"Dangerous keyword '{keyword.upper()}' not allowed."

    if cleaned_query.count('(') != cleaned_query.count(')'): return False, "Unbalanced parentheses"
    return True, "Query validation passed"


def show_advanced_options():
    with st.expander("⚙️ Advanced Options"):
        adv_mode = st.checkbox("Advanced Mode", key='adv_mode_cb')
        val_queries = st.checkbox("Validate Queries", value=True, key='val_q_cb')
        return {'advanced_mode': adv_mode, 'validate_queries': val_queries}


def show_schema_info():
    st.subheader("🗂️ Database Schema")
    schema = get_db_schema()
    if schema:
        with st.expander("View Detailed Schema", expanded=False):
            st.text(get_schema_text(schema))
    else:
        st.error("Could not retrieve database schema.")


def render_sqlbot(llm):
    """Main function to render the SQL Bot interface."""
    st.title("🤖 Advanced SQL Bot")
    st.write("Ask questions involving rankings, groupings, and date logic.")
    init_sql_bot_session_state()
    show_schema_info()
    if llm is None: st.error("❌ No LLM instance available."); return
    schema_text = get_schema_text(get_db_schema())
    advanced_options = show_advanced_options()

    nl_query = st.text_area("Enter your question:", key='sql_bot_nl_query_input',
                            placeholder="e.g., 'Show the top 3 accounts per type by balance'")

    if st.button("🔄 Generate SQL", type="primary", disabled=not nl_query.strip()):
        with st.spinner("🧠 Generating SQL query..."):
            generated_query = generate_sql_from_nl(nl_query, schema_text, llm, advanced_options['advanced_mode'])
            if generated_query:
                st.session_state['sql_bot_generated_query'] = generated_query
                st.success("✅ SQL query generated successfully!")
            else:
                st.error("❌ Failed to generate SQL query.")

    if st.session_state.get('sql_bot_generated_query'):
        edited_query = st.text_area("Review and edit the generated query:", key="sql_editor",
                                    value=st.session_state['sql_bot_generated_query'], height=150)
        is_valid, validation_message = validate_sql_query(edited_query)
        st.info(validation_message)
        if st.button("▶️ Execute Query", type="primary", disabled=not is_valid):
            conn = get_db_connection()
            if conn:
                execute_sql_query(edited_query, conn)
            else:
                st.error("❌ No database connection available.")

    with st.expander("💡 Examples & History"):
        show_example_queries()
        show_query_history()