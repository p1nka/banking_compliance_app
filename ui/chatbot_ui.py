import streamlit as st
import pandas as pd
import re
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import SESSION_CHAT_MESSAGES, SESSION_APP_DF, SESSION_DATA_PROCESSED
from database.connection import get_db_connection
from database.schema import get_db_schema, get_datetime_columns_info
from database.operations import save_sql_query_to_history
from ai.visualizations import auto_visualize
import numpy as np
from datetime import datetime
import logging
import sqlite3
import tempfile
import os

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UploadedDataAnalyzer:
    """Handles analysis of uploaded CSV/Excel data stored in session state."""

    def __init__(self):
        self.data_df = None
        self.schema_info = None
        self.datetime_info = None
        self.temp_db_path = None
        self.initialize_uploaded_data()

    def initialize_uploaded_data(self):
        """Initialize uploaded data from session state."""
        try:
            if SESSION_APP_DF in st.session_state and st.session_state[SESSION_APP_DF] is not None:
                self.data_df = st.session_state[SESSION_APP_DF].copy()
                self.schema_info = self._create_schema_from_dataframe()
                self.datetime_info = self._detect_datetime_columns()
                logger.info(f"Uploaded data initialized: {len(self.data_df)} rows, {len(self.data_df.columns)} columns")
            else:
                logger.info("No uploaded data found in session state")
        except Exception as e:
            logger.error(f"Failed to initialize uploaded data: {str(e)}")
            self.data_df = None

    def _create_schema_from_dataframe(self) -> Dict[str, List[Tuple[str, str]]]:
        """Create schema information from DataFrame."""
        if self.data_df is None or self.data_df.empty:
            return {}

        schema = {"uploaded_data": []}

        for col in self.data_df.columns:
            dtype = str(self.data_df[col].dtype)

            # Map pandas dtypes to SQL-like types
            if 'int' in dtype:
                sql_type = 'INTEGER'
            elif 'float' in dtype:
                sql_type = 'DECIMAL'
            elif 'datetime' in dtype:
                sql_type = 'DATETIME'
            elif 'bool' in dtype:
                sql_type = 'BOOLEAN'
            else:
                sql_type = 'VARCHAR'

            schema["uploaded_data"].append((col, sql_type))

        return schema

    def _detect_datetime_columns(self) -> Dict[str, List[Dict]]:
        """Detect datetime columns in the DataFrame."""
        if self.data_df is None:
            return {}

        datetime_cols = []
        for col in self.data_df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data_df[col]):
                datetime_cols.append({
                    'column': col,
                    'type': 'datetime64',
                    'formatted_type': 'DATETIME'
                })
            elif col.lower().find('date') != -1:
                # Check if it looks like a date column
                sample_values = self.data_df[col].dropna().head(5)
                if not sample_values.empty:
                    try:
                        pd.to_datetime(sample_values.iloc[0])
                        datetime_cols.append({
                            'column': col,
                            'type': 'date_string',
                            'formatted_type': 'DATE'
                        })
                    except:
                        pass

        return {"uploaded_data": datetime_cols} if datetime_cols else {}

    def execute_pandas_query(self, query: str) -> pd.DataFrame:
        """Execute a pandas-based query on uploaded data."""
        if self.data_df is None or self.data_df.empty:
            raise Exception("No uploaded data available")

        # Create a temporary SQLite database in memory
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            self.temp_db_path = tmp_file.name

        try:
            # Connect to SQLite and load data
            conn = sqlite3.connect(self.temp_db_path)

            # Write DataFrame to SQLite
            self.data_df.to_sql('uploaded_data', conn, if_exists='replace', index=False)

            # Execute the query (replace any hardcoded table names)
            cleaned_query = self._adapt_query_for_uploaded_data(query)

            with st.status("🔍 Executing Query on Uploaded Data...", state="running", expanded=True) as status:
                st.code(cleaned_query, language='sql')
                start_time = datetime.now()

                try:
                    results_df = pd.read_sql_query(cleaned_query, conn)
                    execution_time = (datetime.now() - start_time).total_seconds()

                    status.update(
                        label=f"✅ Query completed in {execution_time:.2f}s, found {len(results_df)} rows.",
                        state="complete"
                    )

                    logger.info(f"Pandas query executed successfully. Rows: {len(results_df)}")

                except Exception as e:
                    execution_time = (datetime.now() - start_time).total_seconds()
                    status.update(
                        label=f"❌ Query failed after {execution_time:.2f}s",
                        state="error"
                    )
                    raise Exception(f"Query execution failed: {str(e)}")

                finally:
                    conn.close()

            return results_df

        finally:
            # Clean up temporary database
            if os.path.exists(self.temp_db_path):
                try:
                    os.unlink(self.temp_db_path)
                except:
                    pass

    def _adapt_query_for_uploaded_data(self, query: str) -> str:
        """
        Adapt SQL query to work with the uploaded data table name.
        The complex dialect translation is removed as the AI now generates SQLite-native queries.
        """
        # Replace common hardcoded table names with 'uploaded_data'
        hardcoded_tables = ['accounts_data', 'final_comprehensive_data', 'customer_data', 'transactions']

        adapted_query = query
        for table_name in hardcoded_tables:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(table_name), re.IGNORECASE)
            adapted_query = pattern.sub('uploaded_data', adapted_query)

        # FIX: Handle TOP to LIMIT conversion, as LLM might still generate TOP out of habit.
        # This is more robust than a simple `sub` as it moves the LIMIT to the end.
        top_match = re.search(r'SELECT\s+(?:DISTINCT\s+)?TOP\s+(\d+)', adapted_query, re.IGNORECASE)
        if top_match:
            limit_val = top_match.group(1)
            # Remove the "TOP N" part from the query
            adapted_query = re.sub(r'TOP\s+\d+', '', adapted_query, count=1, flags=re.IGNORECASE).strip()
            # Add the "LIMIT N" clause at the end
            if "LIMIT" not in adapted_query.upper():
                adapted_query += f" LIMIT {limit_val}"

        logger.info(f"Original query: {query}")
        logger.info(f"Adapted query for SQLite: {adapted_query}")

        return adapted_query

    def get_schema_for_prompt(self) -> str:
        """Format uploaded data schema for LLM prompt."""
        if not self.schema_info:
            return "No uploaded data schema available."

        schema_text = "Uploaded Data Schema:\n"
        for table_name, columns in self.schema_info.items():
            schema_text += f"Table: {table_name}\n"
            for column_name, column_type in columns:
                schema_text += f"  - {column_name}: {column_type}\n"
            schema_text += "\n"

        # Add datetime column information
        if self.datetime_info:
            schema_text += "DateTime Columns Information:\n"
            for table_name, datetime_cols in self.datetime_info.items():
                if datetime_cols:
                    schema_text += f"Table {table_name}:\n"
                    for col_info in datetime_cols:
                        schema_text += f"  - {col_info['column']}: {col_info['type']}\n"
                    schema_text += "\n"

        return schema_text


class DatabaseSQLAnalyzer:
    """Handles all direct database interactions, including connection, schema retrieval, and query execution."""

    def __init__(self):
        self.connection = None
        self.schema_info = None
        self.datetime_info = None
        self.primary_table = None
        self.initialize_database()

    def initialize_database(self):
        """Initializes the database connection and retrieves the schema."""
        try:
            self.connection = get_db_connection()
            if self.connection:
                self.schema_info = get_db_schema()
                self.datetime_info = get_datetime_columns_info()
                if self.schema_info:
                    self.primary_table = self._determine_primary_table()
                logger.info(f"Database initialized successfully. Primary table: {self.primary_table}")
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            st.error(f"❌ Database initialization failed: {str(e)}")
            self.connection = None
            self.schema_info = None

    def _determine_primary_table(self) -> Optional[str]:
        """Heuristically determines the most important table in the schema."""
        if not self.schema_info:
            return None
        priority_tables = ['final_comprehensive_data', 'sales', 'orders', 'customers', 'transactions']
        for table in priority_tables:
            if table in self.schema_info:
                return table
        return list(self.schema_info.keys())[0] if self.schema_info else None

    def execute_sql(self, query: str) -> pd.DataFrame:
        """Executes a SQL query and returns a pandas DataFrame with robust datetime handling."""
        if not self.connection:
            raise Exception("Database connection not available")

        cleaned_query = self._clean_and_validate_query(query)
        logger.info(f"Executing query: {cleaned_query}")

        with st.status("🔍 Executing SQL Query...", state="running", expanded=True) as status:
            st.code(cleaned_query, language='sql')
            start_time = datetime.now()

            try:
                # Get all datetime columns from database info
                datetime_columns = self._get_datetime_columns_for_query(cleaned_query)
                logger.info(f"Detected datetime columns: {datetime_columns}")

                # Execute query with datetime parsing
                if datetime_columns:
                    results_df = pd.read_sql(
                        cleaned_query,
                        self.connection,
                        parse_dates=datetime_columns
                    )
                else:
                    results_df = pd.read_sql(cleaned_query, self.connection)

                # Post-process datetime columns that might have been missed
                results_df = self._post_process_datetime_columns(results_df)

                execution_time = (datetime.now() - start_time).total_seconds()
                status.update(
                    label=f"✅ Query completed in {execution_time:.2f}s, found {len(results_df)} rows.",
                    state="complete"
                )

                logger.info(
                    f"Query executed successfully. Rows: {len(results_df)}, Columns: {list(results_df.columns)}")

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                status.update(
                    label=f"❌ Query failed after {execution_time:.2f}s",
                    state="error"
                )
                logger.error(f"SQL execution failed: {str(e)}")
                raise Exception(f"SQL execution failed: {str(e)}")

        # Save query to history
        try:
            save_sql_query_to_history("AI Assistant Query", cleaned_query, execution_time)
        except Exception as e:
            logger.warning(f"Could not save query to history: {e}")
            st.warning(f"Could not save query to history: {e}")

        return results_df

    def _get_datetime_columns_for_query(self, query: str) -> List[str]:
        """Extract datetime columns that are likely to be in the query results."""
        datetime_columns = []

        if not self.datetime_info:
            return datetime_columns

        query_lower = query.lower()

        for table_name, table_datetime_cols in self.datetime_info.items():
            for col_info in table_datetime_cols:
                col_name = col_info['column']
                # Check if column is referenced in the query
                if col_name.lower() in query_lower:
                    datetime_columns.append(col_name)

        return datetime_columns

    def _post_process_datetime_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process DataFrame to ensure datetime columns are properly converted."""
        if df.empty:
            return df

        for col in df.columns:
            # Check if column contains datetime-like strings
            if df[col].dtype == 'object' and not df[col].empty:
                # Sample the first non-null value
                sample_value = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if sample_value and self._looks_like_datetime(str(sample_value)):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Converted column '{col}' to datetime")
                    except Exception as e:
                        logger.warning(f"Failed to convert column '{col}' to datetime: {e}")

        return df

    def _looks_like_datetime(self, value: str) -> bool:
        """Check if a string value looks like a datetime."""
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # YYYY-MM-DD HH:MM:SS
        ]

        for pattern in datetime_patterns:
            if re.search(pattern, value):
                return True
        return False

    def _clean_and_validate_query(self, query: str) -> str:
        """Basic cleaning and validation to prevent harmful SQL commands."""
        if not query or not isinstance(query, str):
            raise Exception("Generated query is empty or not a string.")

        # Clean up the query
        query = query.strip().rstrip(';')

        # Remove markdown code blocks if present
        query = re.sub(r'^```sql\s*|\s*```$', '', query.strip())
        query = re.sub(r'^```\s*|\s*```$', '', query.strip())

        if not query:
            raise Exception("Query is empty after cleaning.")

        # Security validation
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC']
        query_upper = query.upper()

        if not query_upper.strip().startswith('SELECT'):
            raise Exception(
                f"The AI failed to generate a valid SQL query. Query must start with SELECT. Got: {query[:100]}...")

        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {query_upper} ' or query_upper.startswith(f'{keyword} '):
                raise Exception(f"Dangerous SQL keyword '{keyword}' is not allowed.")

        return query

    def get_schema_for_prompt(self) -> str:
        """Formats the database schema as a string for the LLM prompt."""
        if not self.schema_info:
            return "No database schema available."

        schema_text = "Database Schema:\n"
        for table_name, columns in self.schema_info.items():
            schema_text += f"Table: {table_name}\n"
            for column_name, column_type in columns:
                schema_text += f"  - {column_name}: {column_type}\n"
            schema_text += "\n"

        # Add datetime column information
        if self.datetime_info:
            schema_text += "DateTime Columns Information:\n"
            for table_name, datetime_cols in self.datetime_info.items():
                if datetime_cols:
                    schema_text += f"Table {table_name}:\n"
                    for col_info in datetime_cols:
                        schema_text += f"  - {col_info['column']}: {col_info['type']}\n"
                    schema_text += "\n"

        return schema_text


class AI_Data_Assistant:
    """A unified AI assistant that can query a database, generate insights, and create visualizations."""

    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.sql_analyzer = DatabaseSQLAnalyzer()
        self.uploaded_analyzer = UploadedDataAnalyzer()
        self.data_source = self._determine_data_source()
        self.setup_chains()

    def _determine_data_source(self) -> str:
        """Determine whether to use uploaded data or database."""
        # Priority: uploaded data first, then database
        if (SESSION_APP_DF in st.session_state and
                st.session_state[SESSION_APP_DF] is not None and
                not st.session_state[SESSION_APP_DF].empty):
            return "uploaded"
        elif self.sql_analyzer.connection is not None:
            return "database"
        else:
            return "none"

    def setup_chains(self):
        """Sets up the LangChain chains for SQL generation and text summarization."""
        if not self.llm_model:
            logger.error("LLM model not provided")
            return

        sql_generation_template = """You are an expert SQL analyst. Your task is to convert natural language questions into precise, executable SQL queries for the specified SQL dialect.

            {schema_and_rules}

            CRITICAL INSTRUCTIONS:
            1. Generate ONLY SELECT statements.
            2. Use the EXACT table and column names provided in the schema.
            3. Use the correct date functions for the target SQL dialect.
            4. Return ONLY the SQL query - no markdown, no explanations, no extra text.
            5. If you cannot generate a valid query, return: SELECT 1 as no_results WHERE 1=0

            Question: {question}
            SQL Query:"""

        sql_generation_prompt = ChatPromptTemplate.from_template(sql_generation_template)

        self.sql_generation_chain = (
                RunnablePassthrough.assign(schema_and_rules=lambda _: self._get_schema_and_rules_for_prompt())
                | sql_generation_prompt
                | self.llm_model
                | StrOutputParser()
        )

        insight_generation_prompt = ChatPromptTemplate.from_messages([
            ("system",
             """You are a helpful data analyst. Based on the user's question and query results, provide a concise, insightful summary of the key findings. 

             Format your response as:
             1. Direct answer to the question (1-2 sentences)
             2. Key insights from the data (2-3 bullet points)
             3. Any notable patterns or recommendations (if applicable)

             Keep it concise but informative."""),
            ("human",
             "Question: '{question}'\nQuery returned {row_count} rows.\nData preview: {data_preview}\n\nPlease summarize the findings.")
        ])

        self.insight_generation_chain = (
                insight_generation_prompt
                | self.llm_model
                | StrOutputParser()
        )

    def _create_sqlite_aware_prompt(self) -> str:
        """Builds a set of instructions for handling datetime queries in SQLite."""
        return """
SQL DIALECT: SQLite

DATETIME GUIDELINES FOR SQLite:
- Use `datetime('now')` for the current timestamp. Use `date('now')` for the current date.
- Use `strftime('%Y', date_column)` to get the year. `strftime('%m', ...)` for month.
- To subtract time, use modifiers: `datetime('now', '-1 year')`, `date('now', 'start of month', '-1 month')`.
- For date ranges: `WHERE date_column BETWEEN '2023-01-01' AND '2023-12-31'`

TEMPORAL KEYWORDS MAPPING (SQLite):
- "this year" -> `WHERE strftime('%Y', date_column) = strftime('%Y', 'now')`
- "last year" -> `WHERE strftime('%Y', date_column) = CAST(strftime('%Y', 'now') AS INTEGER) - 1`
- "this month" -> `WHERE strftime('%Y-%m', date_column) = strftime('%Y-%m', 'now')`
- "last month" -> `WHERE date(date_column) BETWEEN date('now', 'start of month', '-1 month') AND date('now', 'start of month', '-1 day')`
- "last 30 days" -> `WHERE date(date_column) >= date('now', '-30 days')`
- "today" -> `WHERE date(date_column) = date('now')`
"""

    def _create_sql_server_aware_prompt(self) -> str:
        """Builds a detailed set of instructions for handling datetime queries in SQL Server."""
        return """
SQL DIALECT: SQL Server

DATETIME GUIDELINES FOR SQL SERVER:
- Use `GETDATE()` for the current date/time.
- Use `DATEADD(month, -1, GETDATE())` to subtract time.
- Use `YEAR(date_column)`, `MONTH(date_column)`.

TEMPORAL KEYWORDS MAPPING (SQL Server):
- "this year" -> `WHERE YEAR(date_column) = YEAR(GETDATE())`
- "last year" -> `WHERE YEAR(date_column) = YEAR(GETDATE()) - 1`
- "this month" -> `WHERE YEAR(date_column) = YEAR(GETDATE()) AND MONTH(date_column) = MONTH(GETDATE())`
- "last month" -> `WHERE date_column >= DATEADD(MONTH, -1, DATEADD(DAY, 1 - DAY(GETDATE()), GETDATE())) AND date_column < DATEADD(DAY, 1 - DAY(GETDATE()), GETDATE())`
- "last 30 days" -> `WHERE date_column >= DATEADD(DAY, -30, GETDATE())`
- "today" -> `WHERE CAST(date_column AS DATE) = CAST(GETDATE() AS DATE)`
"""

    def _get_business_rules_prompt(self) -> str:
        """Defines specific business logic for complex queries."""
        business_rules = """
BUSINESS RULES:
- "Dormant Account": An account where the `Expected_Account_Dormant` column is 'Yes'.
- "CBUAE Transfer Eligibility": An account where `Freeze_Threshold_Date` is in the past and `Current_Balance` > 0.
"""
        return business_rules

    def _get_schema_and_rules_for_prompt(self):
        """Constructs the full schema and rules text, dynamically selecting the SQL dialect guidance."""
        if self.data_source == "uploaded":
            schema_text = self.uploaded_analyzer.get_schema_for_prompt()
            dialect_rules = self._create_sqlite_aware_prompt()
        else:
            schema_text = self.sql_analyzer.get_schema_for_prompt()
            dialect_rules = self._create_sql_server_aware_prompt()

        business_rules = self._get_business_rules_prompt()
        return f"{dialect_rules}\n{schema_text}\n{business_rules}"

    def process_query(self, user_query: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Main processing pipeline: NL -> SQL -> Data -> Insight -> (Hand off to viz)."""
        try:
            # Re-determine data source in case it changed
            self.data_source = self._determine_data_source()

            if self.data_source == "none":
                return "❌ **No data available.** Please upload data or connect to a database first.", None

            # Generate SQL
            with st.spinner("🧠 Generating SQL..."):
                sql_query = self._generate_sql(user_query)
                logger.info(f"Generated SQL for {self.data_source} source: {sql_query}")

            # Execute SQL based on data source
            if self.data_source == "uploaded":
                query_results = self.uploaded_analyzer.execute_pandas_query(sql_query)
            else:
                query_results = self.sql_analyzer.execute_sql(sql_query)

            # Generate insights
            if not query_results.empty:
                with st.spinner("💡 Generating insights..."):
                    text_insights = self._generate_text_insights(user_query, query_results)
            else:
                text_insights = "The query ran successfully but returned no data. Your criteria might be too specific, or there might be no records matching your request."

            return text_insights, query_results

        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            error_message = f"❌ **Error processing your query:** {str(e)}\n\nPlease try rephrasing your question or check if your request is supported."
            return error_message, None

    def _generate_sql(self, question: str) -> str:
        """Generates a SQL query from a natural language question."""
        try:
            sql_query = self.sql_generation_chain.invoke({"question": question})
            logger.info(f"Raw SQL from LLM: {sql_query}")

            # Clean up the query more thoroughly
            cleaned_query = re.sub(r'^```sql\s*|\s*```$', '', sql_query.strip())
            cleaned_query = re.sub(r'^```\s*|\s*```$', '', cleaned_query.strip())

            if not cleaned_query:
                raise Exception("LLM returned empty query")

            return cleaned_query

        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise Exception(f"Failed to generate SQL query: {str(e)}")

    def _generate_text_insights(self, question: str, results: pd.DataFrame) -> str:
        """Generates a text summary of the query results."""
        try:
            # Create a preview of the data for context
            data_preview = ""
            if not results.empty:
                # Show first few rows or summary statistics
                if len(results) <= 5:
                    data_preview = results.to_string(index=False)
                else:
                    data_preview = f"First 3 rows:\n{results.head(3).to_string(index=False)}\n\nColumns: {list(results.columns)}"

            insights = self.insight_generation_chain.invoke({
                "question": question,
                "row_count": len(results),
                "data_preview": data_preview[:500]  # Limit preview size
            })

            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return f"Query returned {len(results)} rows. The data is ready for analysis, but I couldn't generate detailed insights due to an error: {str(e)}"


def display_chat_interface(llm_model):
    """Renders the main Streamlit chat interface for the AI Data Assistant."""
    try:
        if "assistant" not in st.session_state:
            with st.spinner("🚀 Initializing AI Assistant..."):
                st.session_state.assistant = AI_Data_Assistant(llm_model)

        assistant = st.session_state.assistant

        # Check system status and show appropriate message
        if not assistant.llm_model:
            st.error("🔴 **AI Model Not Loaded.** Please check your API key and restart the app.")
            return

        # Determine current data source and show status
        current_data_source = assistant._determine_data_source()

        if current_data_source == "uploaded":
            uploaded_df = st.session_state.get(SESSION_APP_DF)
            if uploaded_df is not None:
                st.success(
                    f"🟢 **System Ready** - Using uploaded data ({len(uploaded_df)} rows, {len(uploaded_df.columns)} columns)")
            else:
                st.warning("⚠️ **Uploaded data reference lost.** Please re-upload your data.")
                return
        elif current_data_source == "database":
            st.success("🟢 **System Ready** - Using database connection")
        else:
            st.warning("⚠️ **No data source available.** Please upload data or connect to a database.")
            return

    except Exception as e:
        logger.error(f"Fatal error during assistant initialization: {e}")
        st.error(f"❌ **Initialization Error:** {e}")
        return

    # Initialize Chat History with dynamic message
    if SESSION_CHAT_MESSAGES not in st.session_state:
        data_source_msg = "uploaded data" if current_data_source == "uploaded" else "database"
        st.session_state[SESSION_CHAT_MESSAGES] = [{
            "role": "assistant",
            "content": f"Hello! I'm your AI Data Assistant. I can help you analyze your {data_source_msg} by answering questions in natural language. What would you like to know about your data?"
        }]

    # Display Chat History
    for message in st.session_state[SESSION_CHAT_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display results if available
            if "results_df" in message and message["results_df"] is not None:
                results_df = message["results_df"]
                if not results_df.empty:
                    # Show visualization
                    try:
                        auto_visualize(results_df)
                    except Exception as e:
                        logger.warning(f"Visualization failed: {e}")
                        st.warning("Could not generate visualization for this data.")

                    # Show raw data in expander
                    with st.expander(f"📊 View Raw Data ({len(results_df)} rows)"):
                        st.dataframe(results_df, use_container_width=True)
                else:
                    st.info("Query returned no results.")

    # Example Questions Expander
    with st.expander("💡 Example Questions"):
        st.markdown("""
        **Basic Queries:**
        - "How many accounts are there in total?"
        - "Show me the 10 accounts with the highest balance"
        - "What is the average account balance?"

        **Date-based Queries:**
        - "Show me accounts created in the last 6 months"
        - "What is the monthly trend of account creation this year?"
        - "How many accounts were created in January 2024?"
        - "Which accounts were opened last year?"
        - "Show me activity from last month"

        **Business-specific Queries:**
        - "Show me accounts that are eligible for CBUAE transfer"
        - "Which accounts are dormant?"
        - "Show me accounts with balances over $10,000"
        """)

    # User Input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message
        st.session_state[SESSION_CHAT_MESSAGES].append({"role": "user", "content": prompt})

        # Process and add assistant response
        with st.chat_message("assistant"):
            try:
                response_text, results_df = assistant.process_query(prompt)

                # Display the response
                st.markdown(response_text)

                # Handle results
                if results_df is not None and not results_df.empty:
                    # Show visualization
                    try:
                        auto_visualize(results_df)
                    except Exception as e:
                        logger.warning(f"Visualization failed: {e}")
                        st.warning("Could not generate visualization for this data.")

                    # Show raw data in expander
                    with st.expander(f"📊 View Raw Data ({len(results_df)} rows)"):
                        st.dataframe(results_df, use_container_width=True)

                elif results_df is not None and results_df.empty:
                    st.info("Query executed successfully but returned no results.")

                # Save assistant response
                assistant_response = {
                    "role": "assistant",
                    "content": response_text,
                    "results_df": results_df
                }
                st.session_state[SESSION_CHAT_MESSAGES].append(assistant_response)

            except Exception as e:
                error_msg = f"❌ **An error occurred:** {str(e)}"
                st.error(error_msg)
                logger.error(f"Chat interface error: {str(e)}")

                # Save error message
                st.session_state[SESSION_CHAT_MESSAGES].append({
                    "role": "assistant",
                    "content": error_msg
                })

        # Rerun to show the new messages
        st.rerun()


def render_chatbot(llm_model):
    """Main entry point for the chatbot UI."""
    return display_chat_interface(llm_model)