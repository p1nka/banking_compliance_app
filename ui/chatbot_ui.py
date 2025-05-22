import streamlit as st
import pandas as pd
import re
import json
import pyodbc
from typing import Dict, Any, Optional, Tuple, List
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import AgentAction, AgentFinish
from config import SESSION_CHAT_MESSAGES, SESSION_COLUMN_MAPPING
from ai.llm import generate_sql
from database.connection import get_db_connection
from database.schema import get_db_schema
from database.operations import save_sql_query_to_history
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime


class DatabaseSQLAnalyzer:
    """Enhanced SQL analyzer using your existing database infrastructure"""

    def __init__(self):
        self.connection = None
        self.schema_info = None
        self.primary_table = None
        self.initialize_database()

    def initialize_database(self):
        """Initialize database connection and schema"""
        try:
            # Use your existing connection function
            self.connection = get_db_connection()

            if self.connection:
                st.success("✅ Connected to Azure SQL Database successfully!")

                # Get schema using your existing function
                self.schema_info = get_db_schema()

                if self.schema_info:
                    st.success(f"✅ Database schema loaded: {len(self.schema_info)} tables found")
                    self.primary_table = self._determine_primary_table()

                    # Display available tables
                    with st.expander("📊 Available Database Tables", expanded=False):
                        for table_name, columns in self.schema_info.items():
                            st.write(f"**{table_name}** ({len(columns)} columns)")
                            col_info = [f"{col[0]} ({col[1]})" for col in columns[:5]]
                            if len(columns) > 5:
                                col_info.append(f"... and {len(columns) - 5} more")
                            st.write(f"   • {', '.join(col_info)}")
                else:
                    st.warning("⚠️ Could not load database schema")
            else:
                st.error("❌ Failed to connect to database")

        except Exception as e:
            st.error(f"❌ Database initialization failed: {str(e)}")
            self.connection = None
            self.schema_info = None

    def _determine_primary_table(self) -> str:
        """Determine the primary table for analysis"""
        if not self.schema_info:
            return None

        # Priority order for banking compliance tables
        priority_tables = [
            'accounts_data',
            'dormant_flags',
            'dormant_ledger',
            'analysis_results',
            'sql_query_history'
        ]

        # Check for preferred tables first
        for table in priority_tables:
            if table in self.schema_info:
                return table

        # Return the first available table
        return list(self.schema_info.keys())[0] if self.schema_info else None

    def execute_sql(self, query: str, save_to_history: bool = True) -> pd.DataFrame:
        """Execute SQL query using your database connection"""
        try:
            if not self.connection:
                raise Exception("Database connection not available")

            # Clean and validate the query
            cleaned_query = self._clean_and_validate_query(query)

            st.info(f"🔍 **Executing SQL on Azure Database:**")
            st.code(cleaned_query, language='sql')

            # Execute query using pandas
            start_time = datetime.now()
            result_df = pd.read_sql(cleaned_query, self.connection)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            st.success(f"✅ Query completed in {execution_time:.0f}ms - Found {len(result_df)} records")

            # Save to history if enabled
            if save_to_history:
                try:
                    # This will use your existing save function
                    save_sql_query_to_history("Chatbot Query", cleaned_query)
                except Exception as e:
                    st.warning(f"Could not save to history: {e}")

            # Show preview of results
            if not result_df.empty:
                with st.expander("📋 Query Results Preview", expanded=False):
                    st.dataframe(result_df.head(20), use_container_width=True)
                    if len(result_df) > 20:
                        st.info(f"Showing first 20 of {len(result_df)} total records")

            return result_df

        except Exception as e:
            error_msg = f"SQL execution error: {str(e)}"
            st.error(f"❌ {error_msg}")
            raise Exception(error_msg)

    def _clean_and_validate_query(self, query: str) -> str:
        """Clean and validate SQL query for your database"""
        # Remove dangerous SQL commands (using your existing approach)
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'EXEC']
        query_upper = query.upper()

        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {query_upper} ' or query_upper.startswith(f'{keyword} '):
                raise Exception(f"Dangerous SQL keyword '{keyword}' not allowed")

        # Ensure query is a SELECT statement
        if not query_upper.strip().startswith('SELECT'):
            raise Exception("Only SELECT statements are allowed")

        # Replace generic table references with actual table names
        if self.primary_table and 'data_table' in query.lower():
            query = re.sub(r'\bdata_table\b', self.primary_table, query, flags=re.IGNORECASE)

        return query.strip()

    def get_schema_description(self) -> str:
        """Get comprehensive schema description for LLM context"""
        if not self.schema_info:
            return "No database schema available"

        description = []
        description.append("AZURE SQL DATABASE SCHEMA:")
        description.append("=" * 50)

        for table_name, columns in self.schema_info.items():
            description.append(f"\nTABLE: {table_name}")
            description.append("COLUMNS:")

            for col_name, col_type in columns:
                description.append(f"  - {col_name} ({col_type})")

            # Add sample data context for key tables
            if table_name == 'accounts_data':
                description.append("  * Main banking accounts table")
                description.append("  * Key fields: Account_ID, Customer_ID, Account_Type, Current_Balance")
                description.append("  * Date fields for tracking dormancy and activity")
            elif table_name == 'dormant_flags':
                description.append("  * Dormant account flags and instructions")
            elif table_name == 'insight_log':
                description.append("  * Analysis insights and observations")

        description.append(f"\nPRIMARY TABLE FOR ANALYSIS: {self.primary_table}")
        description.append("\nNOTE: Use exact table and column names as shown above")

        return "\n".join(description)

    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a specific table"""
        try:
            if not self.connection:
                return 0

            cursor = self.connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except:
            return 0


class DynamicVisualizationEngine:
    """Enhanced visualization engine for banking compliance data"""

    def __init__(self):
        self.chart_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Banking-specific color schemes
        self.banking_colors = {
            'risk_levels': ['#28a745', '#ffc107', '#fd7e14', '#dc3545'],  # Green to Red
            'account_types': ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1'],
            'status': ['#28a745', '#6c757d', '#dc3545']  # Active, Inactive, Closed
        }

    def analyze_and_visualize(self, query_result: pd.DataFrame, query: str, original_question: str) -> Tuple[Any, str]:
        """Analyze SQL results and create banking-focused visualizations"""

        if query_result.empty:
            return None, "❌ Query returned no data to visualize"

        st.info(f"📊 Creating banking compliance visualization for {len(query_result)} records...")

        try:
            # Analyze the result structure
            analysis = self._analyze_banking_data_structure(query_result)

            # Determine best visualization for banking data
            chart_config = self._determine_banking_chart_type(analysis, original_question)

            # Generate the chart with banking context
            chart = self._create_banking_chart(query_result, chart_config)

            if chart is None:
                return None, "❌ Failed to create visualization"

            # Generate banking-specific insights
            insight = self._generate_banking_insights(query_result, chart_config, original_question)

            st.success("✅ Banking compliance visualization created!")
            return chart, insight

        except Exception as e:
            error_msg = f"❌ Visualization error: {str(e)}"
            st.error(error_msg)
            return None, error_msg

    def _analyze_banking_data_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze banking data structure with domain knowledge"""
        analysis = {
            'num_rows': len(df),
            'num_cols': len(df.columns),
            'columns': list(df.columns),
            'numeric_columns': [],
            'categorical_columns': [],
            'date_columns': [],
            'boolean_columns': [],
            'banking_context': {}
        }

        # Identify column types and banking context
        for col in df.columns:
            col_lower = col.lower()

            if pd.api.types.is_numeric_dtype(df[col]):
                analysis['numeric_columns'].append(col)

                # Identify banking-specific numeric fields
                if any(term in col_lower for term in ['balance', 'amount', 'charges']):
                    analysis['banking_context']['financial'] = analysis['banking_context'].get('financial', []) + [col]
                elif any(term in col_lower for term in ['count', 'total']):
                    analysis['banking_context']['metrics'] = analysis['banking_context'].get('metrics', []) + [col]

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis['date_columns'].append(col)

                # Identify banking date fields
                if any(term in col_lower for term in ['dormancy', 'maturity', 'creation', 'last_activity']):
                    analysis['banking_context']['timeline'] = analysis['banking_context'].get('timeline', []) + [col]

            elif pd.api.types.is_bool_dtype(df[col]) or df[col].dtype == 'object' and set(
                    df[col].dropna().unique()).issubset({'Yes', 'No', 'Unknown', True, False}):
                analysis['boolean_columns'].append(col)

            else:
                analysis['categorical_columns'].append(col)

                # Identify banking categories
                if any(term in col_lower for term in ['account_type', 'customer_type', 'status']):
                    analysis['banking_context']['categories'] = analysis['banking_context'].get('categories', []) + [
                        col]
                elif any(term in col_lower for term in ['id', 'number']):
                    analysis['banking_context']['identifiers'] = analysis['banking_context'].get('identifiers', []) + [
                        col]

        return analysis

    def _determine_banking_chart_type(self, analysis: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Determine optimal chart type for banking compliance data"""

        config = {
            'chart_type': 'bar',
            'x_column': None,
            'y_column': None,
            'color_column': None,
            'names_column': None,
            'values_column': None,
            'title': self._generate_banking_title(question),
            'color_scheme': 'default'
        }

        question_lower = question.lower()
        banking_context = analysis.get('banking_context', {})

        # Banking-specific chart type decisions
        if 'dormant' in question_lower or 'compliance' in question_lower:
            config['color_scheme'] = 'risk_levels'

        elif 'account' in question_lower and 'type' in question_lower:
            config['color_scheme'] = 'account_types'

        # Standard chart type logic enhanced for banking
        if analysis['num_cols'] == 1:
            col = analysis['columns'][0]
            if col in analysis['numeric_columns']:
                config['chart_type'] = 'histogram'
                config['x_column'] = col
            else:
                config['chart_type'] = 'pie'
                config['names_column'] = col

        elif analysis['num_cols'] == 2:
            if len(analysis['categorical_columns']) == 1 and len(analysis['numeric_columns']) == 1:
                cat_col = analysis['categorical_columns'][0]
                num_col = analysis['numeric_columns'][0]

                # Special handling for banking metrics
                if 'balance' in num_col.lower() or 'amount' in num_col.lower():
                    config['chart_type'] = 'bar'
                    config['x_column'] = cat_col
                    config['y_column'] = num_col
                    config['color_scheme'] = 'banking'
                elif analysis['num_rows'] <= 10:
                    config['chart_type'] = 'pie'
                    config['names_column'] = cat_col
                    config['values_column'] = num_col
                else:
                    config['chart_type'] = 'bar'
                    config['x_column'] = cat_col
                    config['y_column'] = num_col

            elif len(analysis['date_columns']) == 1 and len(analysis['numeric_columns']) == 1:
                config['chart_type'] = 'line'
                config['x_column'] = analysis['date_columns'][0]
                config['y_column'] = analysis['numeric_columns'][0]

        # Override based on banking keywords
        if 'trend' in question_lower or 'over time' in question_lower:
            if analysis['date_columns']:
                config['chart_type'] = 'line'
                config['x_column'] = analysis['date_columns'][0]
                if analysis['numeric_columns']:
                    config['y_column'] = analysis['numeric_columns'][0]

        elif 'distribution' in question_lower and analysis['categorical_columns']:
            config['chart_type'] = 'pie'
            config['names_column'] = analysis['categorical_columns'][0]

        return config

    def _create_banking_chart(self, df: pd.DataFrame, config: Dict[str, Any]) -> Any:
        """Create banking-focused Plotly charts"""

        try:
            chart_type = config['chart_type']
            colors = self._get_color_scheme(config['color_scheme'])

            common_params = {
                'title': config['title'],
                'color_discrete_sequence': colors
            }

            if chart_type == 'bar':
                chart = px.bar(
                    df,
                    x=config.get('x_column'),
                    y=config.get('y_column'),
                    color=config.get('color_column'),
                    **common_params
                )

                # Add value labels for banking data
                chart.update_traces(
                    texttemplate='%{y:,.0f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Value: %{y:,.2f}<extra></extra>'
                )

            elif chart_type == 'pie':
                if config.get('values_column'):
                    chart = px.pie(
                        df,
                        names=config.get('names_column'),
                        values=config.get('values_column'),
                        **common_params
                    )
                else:
                    names_col = config.get('names_column')
                    if names_col and names_col in df.columns:
                        value_counts = df[names_col].value_counts()
                        chart = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            **common_params
                        )

                # Enhance pie chart for banking data
                chart.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )

            elif chart_type == 'line':
                chart = px.line(
                    df,
                    x=config.get('x_column'),
                    y=config.get('y_column'),
                    color=config.get('color_column'),
                    **common_params,
                    markers=True
                )

                # Banking time series enhancements
                chart.update_traces(
                    line=dict(width=3),
                    marker=dict(size=8),
                    hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:,.2f}<extra></extra>'
                )

            elif chart_type == 'scatter':
                chart = px.scatter(
                    df,
                    x=config.get('x_column'),
                    y=config.get('y_column'),
                    color=config.get('color_column'),
                    **common_params,
                    size_max=60
                )

            else:
                # Default banking bar chart
                if len(df.columns) >= 2:
                    chart = px.bar(df, x=df.columns[0], y=df.columns[1], **common_params)
                else:
                    chart = px.bar(df, y=df.columns[0], **common_params)

            # Banking-specific styling
            chart.update_layout(
                plot_bgcolor='rgba(248, 249, 250, 0.8)',
                paper_bgcolor='white',
                font=dict(size=12, family="Arial, sans-serif"),
                title_font_size=18,
                title_font_color='#2c3e50',
                showlegend=True if config.get('color_column') else False,
                height=500,
                margin=dict(t=80, b=60, l=60, r=60)
            )

            # Banking compliance styling
            chart.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(size=14, color='#34495e')
            )
            chart.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                title_font=dict(size=14, color='#34495e')
            )

            return chart

        except Exception as e:
            st.error(f"❌ Error creating banking chart: {e}")
            return None

    def _get_color_scheme(self, scheme_name: str) -> List[str]:
        """Get appropriate color scheme for banking data"""
        if scheme_name in self.banking_colors:
            return self.banking_colors[scheme_name]
        return self.chart_colors

    def _generate_banking_title(self, question: str) -> str:
        """Generate banking-appropriate chart titles"""
        title = question.strip()
        if title.endswith('?'):
            title = title[:-1]

        # Banking-specific title enhancements
        if 'dormant' in title.lower():
            title = f"Banking Compliance Analysis: {title}"
        elif 'account' in title.lower():
            title = f"Account Analysis: {title}"
        elif 'balance' in title.lower() or 'amount' in title.lower():
            title = f"Financial Analysis: {title}"

        return title[:80] + ('...' if len(title) > 80 else '')

    def _generate_banking_insights(self, df: pd.DataFrame, config: Dict[str, Any], question: str) -> str:
        """Generate banking compliance insights"""

        insights = []
        insights.append(f"📊 **Banking Analysis Results:** {len(df)} records processed")

        # Financial insights for balance/amount columns
        if config.get('y_column'):
            y_col = config['y_column']
            if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                total = df[y_col].sum()
                avg = df[y_col].mean()
                max_val = df[y_col].max()
                min_val = df[y_col].min()

                insights.append(f"💰 **Financial Metrics:**")
                insights.append(f"   • Total Value: ${total:,.2f}")
                insights.append(f"   • Average: ${avg:,.2f}")
                insights.append(f"   • Range: ${min_val:,.2f} to ${max_val:,.2f}")

                # Banking-specific thresholds
                if 'balance' in y_col.lower():
                    low_balance_count = len(df[df[y_col] < 100])
                    if low_balance_count > 0:
                        insights.append(f"⚠️ **Risk Alert:** {low_balance_count} accounts with balance < $100")

                # Top performer analysis
                if config.get('x_column'):
                    x_col = config['x_column']
                    max_idx = df[y_col].idxmax()
                    top_category = df.loc[max_idx, x_col]
                    top_value = df.loc[max_idx, y_col]
                    insights.append(f"🏆 **Top Performer:** {top_category} (${top_value:,.2f})")

        # Compliance insights
        compliance_keywords = ['dormant', 'flag', 'compliance', 'regulatory']
        if any(keyword in question.lower() for keyword in compliance_keywords):
            insights.append("🛡️ **Compliance Note:** Review highlighted items for regulatory requirements")

        # Chart type explanation
        chart_explanations = {
            'bar': 'Bar chart showing categorical comparisons for easy analysis',
            'pie': 'Pie chart displaying distribution percentages across categories',
            'line': 'Line chart tracking trends and patterns over time',
            'scatter': 'Scatter plot revealing relationships between variables'
        }

        chart_type = config.get('chart_type', 'bar')
        if chart_type in chart_explanations:
            insights.append(f"📈 **Visualization:** {chart_explanations[chart_type]}")

        return "\n".join(insights)


class EnhancedBankingChatbot:
    """Enhanced chatbot specifically designed for banking compliance with database integration"""

    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.sql_analyzer = DatabaseSQLAnalyzer()
        self.viz_engine = DynamicVisualizationEngine()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=15  # Remember more for banking context
        )
        self.setup_banking_chains()

    def setup_banking_chains(self):
        """Setup LangChain chains optimized for banking compliance"""

        if not self.llm_model:
            st.warning("⚠️ LLM model not available - chains not initialized")
            return

        # Banking-focused SQL Generation Chain
        banking_sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SQL analyst specializing in banking compliance and dormant account analysis.

Database Schema:
{schema}

BANKING DOMAIN EXPERTISE:
- Focus on account analysis, customer behavior, and compliance requirements
- Understand dormancy periods, maturity dates, and regulatory triggers
- Recognize banking terminology and account lifecycle stages

SQL GENERATION RULES:
1. Always use SELECT statements only
2. Use exact table and column names from schema
3. For banking queries, focus on key metrics: balances, dates, account status
4. Use appropriate date filtering for dormancy analysis
5. Include relevant JOIN conditions for related tables
6. Use descriptive aliases for calculated fields
7. Order results logically (by date, amount, or priority)

BANKING QUERY PATTERNS:
- Dormant accounts: Check Date_Last_Customer_Communication_Any_Type and Date_Last_Cust_Initiated_Activity
- Account analysis: Group by Account_Type, analyze Current_Balance
- Compliance flags: Check Expected_Account_Dormant and related fields
- Customer activity: Look at communication dates and response flags

Generate only the SQL query without explanations.
            """),
            ("human", "{question}")
        ])

        self.banking_sql_chain = (
                RunnablePassthrough.assign(
                    schema=lambda
                        _: self.sql_analyzer.get_schema_description() if self.sql_analyzer else "No schema available"
                )
                | banking_sql_prompt
                | self.llm_model
                | StrOutputParser()
        )

        # Banking Insight Generation Chain
        banking_insight_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a banking compliance expert providing insights on data analysis results.

BANKING EXPERTISE FOCUS:
- Dormant account management and compliance
- Customer communication and activity patterns
- Regulatory requirements and timelines
- Risk assessment and mitigation
- Account lifecycle management

INSIGHT GUIDELINES:
- Provide actionable recommendations
- Highlight compliance risks and requirements
- Suggest follow-up actions for bank staff
- Identify patterns that need attention
- Reference regulatory considerations when relevant

Keep insights professional, concise, and focused on banking operations.
            """),
            ("human",
             "Banking Query: {question}\nData Analysis: {results_summary}\nProvide compliance-focused insights.")
        ])

        self.banking_insight_chain = (
                banking_insight_prompt
                | self.llm_model
                | StrOutputParser()
        )

    def process_banking_query(self, user_query: str) -> Tuple[str, Optional[Any]]:
        """Process banking-specific queries with enhanced context"""

        try:
            st.info(f"🏦 **Processing Banking Query:** '{user_query}'")

            if not self.llm_model:
                return "🚫 AI features are disabled. Please check API key configuration.", None

            if not self.sql_analyzer.connection:
                return "❌ Database connection not available. Please check database configuration.", None

            # Step 1: Generate banking-focused SQL
            st.info("🧠 **Step 1:** Generating banking compliance SQL...")
            sql_query = self._generate_banking_sql(user_query)

            # Step 2: Execute on Azure SQL Database
            st.info("⚡ **Step 2:** Executing query on Azure SQL Database...")
            query_results = self._execute_banking_sql(sql_query)

            # Step 3: Create banking visualization
            st.info("📊 **Step 3:** Creating banking compliance visualization...")
            chart, viz_insights = self.viz_engine.analyze_and_visualize(
                query_results, sql_query, user_query
            )

            # Step 4: Generate banking insights
            st.info("💡 **Step 4:** Generating banking compliance insights...")
            banking_insights = self._generate_banking_insights(user_query, query_results)

            # Step 5: Combine insights
            final_response = self._combine_banking_insights(viz_insights, banking_insights, query_results)

            st.success("✅ **Banking Analysis Complete!**")
            return final_response, chart

        except Exception as e:
            error_msg = f"❌ **Banking Analysis Failed:** {str(e)}"
            st.error(error_msg)

            # Provide banking-specific fallback
            try:
                fallback_response = self._provide_banking_fallback(user_query)
                return fallback_response, None
            except:
                return error_msg, None

    def _generate_banking_sql(self, question: str) -> str:
        """Generate banking-focused SQL queries"""
        try:
            if not self.banking_sql_chain:
                raise Exception("Banking SQL chain not initialized")

            sql_query = self.banking_sql_chain.invoke({"question": question})

            # Clean the SQL query
            sql_query = re.sub(r'^```sql\s*|\s*```$', '', sql_query.strip())
            sql_query = re.sub(r'^```\s*|\s*```$', '', sql_query.strip())

            if not sql_query or not sql_query.upper().startswith('SELECT'):
                sql_query = self._generate_banking_fallback_sql(question)

            return sql_query
        except Exception as e:
            st.warning(f"⚠️ Banking SQL generation failed: {e}. Using fallback.")
            return self._generate_banking_fallback_sql(question)

    def _generate_banking_fallback_sql(self, question: str) -> str:
        """Generate banking-specific fallback SQL queries"""
        question_lower = question.lower()
        primary_table = self.sql_analyzer.primary_table or 'accounts_data'

        # Banking-specific query patterns
        if 'dormant' in question_lower:
            return f"SELECT Account_ID, Customer_ID, Account_Type, Current_Balance, Date_Last_Cust_Initiated_Activity FROM {primary_table} WHERE Expected_Account_Dormant = 'Yes' ORDER BY Date_Last_Cust_Initiated_Activity"
        elif 'balance' in question_lower:
            return f"SELECT Account_Type, AVG(Current_Balance) as Avg_Balance, COUNT(*) as Account_Count FROM {primary_table} GROUP BY Account_Type ORDER BY Avg_Balance DESC"
        elif 'account' in question_lower and 'type' in question_lower:
            return f"SELECT Account_Type, COUNT(*) as Count FROM {primary_table} GROUP BY Account_Type ORDER BY Count DESC"
        elif 'customer' in question_lower:
            return f"SELECT Customer_ID, Account_Type, Current_Balance, Date_Last_Cust_Initiated_Activity FROM {primary_table} ORDER BY Current_Balance DESC"
        elif 'top' in question_lower:
            return f"SELECT TOP 10 Account_ID, Customer_ID, Current_Balance FROM {primary_table} ORDER BY Current_Balance DESC"
        else:
            return f"SELECT TOP 100 * FROM {primary_table} ORDER BY Account_Creation_Date DESC"

    def _execute_banking_sql(self, sql_query: str) -> pd.DataFrame:
        """Execute banking SQL with enhanced error handling"""
        try:
            result_df = self.sql_analyzer.execute_sql(sql_query, save_to_history=True)

            if result_df.empty:
                st.warning("⚠️ Query returned no results - trying broader analysis")
                fallback_query = f"SELECT TOP 50 * FROM {self.sql_analyzer.primary_table}"
                result_df = self.sql_analyzer.execute_sql(fallback_query, save_to_history=False)

            return result_df

        except Exception as e:
            st.error(f"❌ Banking SQL execution failed: {e}")
            # Return empty DataFrame as fallback
            return pd.DataFrame()

    def _generate_banking_insights(self, question: str, results: pd.DataFrame) -> str:
        """Generate banking compliance insights"""
        try:
            if not self.banking_insight_chain or results.empty:
                return self._generate_basic_banking_insights(results)

            # Create banking-focused summary
            results_summary = self._create_banking_summary(results)

            insights = self.banking_insight_chain.invoke({
                "question": question,
                "results_summary": results_summary
            })

            return insights if insights else self._generate_basic_banking_insights(results)

        except Exception as e:
            st.warning(f"⚠️ Banking insight generation failed: {e}")
            return self._generate_basic_banking_insights(results)

    def _create_banking_summary(self, results: pd.DataFrame) -> str:
        """Create banking-focused data summary"""
        summary = []
        summary.append(f"Banking Data Analysis: {len(results)} records")

        # Identify banking-specific columns
        banking_columns = {
            'balance_cols': [col for col in results.columns if 'balance' in col.lower() or 'amount' in col.lower()],
            'date_cols': [col for col in results.columns if 'date' in col.lower()],
            'status_cols': [col for col in results.columns if
                            any(term in col.lower() for term in ['status', 'type', 'flag'])],
            'id_cols': [col for col in results.columns if col.lower().endswith('_id')]
        }

        # Financial summary
        if banking_columns['balance_cols']:
            summary.append("Financial Summary:")
            for col in banking_columns['balance_cols'][:3]:
                if pd.api.types.is_numeric_dtype(results[col]):
                    total = results[col].sum()
                    avg = results[col].mean()
                    summary.append(f"  - {col}: Total=${total:,.2f}, Average=${avg:,.2f}")

        # Account type distribution
        if 'Account_Type' in results.columns:
            type_dist = results['Account_Type'].value_counts()
            summary.append(f"Account Types: {dict(type_dist.head())}")

        # Date range analysis
        date_cols = [col for col in results.columns if
                     'date' in col.lower() and pd.api.types.is_datetime64_any_dtype(results[col])]
        if date_cols:
            for col in date_cols[:2]:
                if results[col].notna().any():
                    min_date = results[col].min()
                    max_date = results[col].max()
                    summary.append(f"{col}: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")

        return "\n".join(summary)

    def _generate_basic_banking_insights(self, results: pd.DataFrame) -> str:
        """Generate basic banking insights as fallback"""
        if results.empty:
            return "📊 **No data found for analysis**"

        insights = []
        insights.append(f"🏦 **Banking Data Summary:** {len(results)} records analyzed")

        # Basic column analysis
        numeric_cols = results.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            insights.append("💰 **Financial Overview:**")
            for col in numeric_cols[:3]:
                if 'balance' in col.lower() or 'amount' in col.lower():
                    total = results[col].sum()
                    avg = results[col].mean()
                    insights.append(f"   • {col}: Total=${total:,.2f}, Average=${avg:,.2f}")

        # Account type analysis
        if 'Account_Type' in results.columns:
            type_counts = results['Account_Type'].value_counts()
            insights.append(f"📋 **Account Types:** {len(type_counts)} different types found")
            for acc_type, count in type_counts.head(3).items():
                insights.append(f"   • {acc_type}: {count} accounts")

        return "\n".join(insights)

    def _combine_banking_insights(self, viz_insights: str, banking_insights: str, results: pd.DataFrame) -> str:
        """Combine banking-focused insights"""
        response_parts = []

        # Banking compliance header
        response_parts.append("🏦 **Banking Compliance Analysis Results**")
        response_parts.append("=" * 50)
        response_parts.append("")

        # Add banking insights first
        if banking_insights and banking_insights.strip():
            response_parts.append("🎯 **Business Intelligence:**")
            response_parts.append(banking_insights)
            response_parts.append("")

        # Add visualization insights
        if viz_insights and viz_insights.strip():
            response_parts.append(viz_insights)
            response_parts.append("")

        # Add compliance notes
        response_parts.append("📊 **Data Quality Check:**")
        response_parts.append(f"   • Records Processed: {len(results):,}")
        response_parts.append(f"   • Data Fields: {len(results.columns)}")

        # Add regulatory reminder
        response_parts.append("")
        response_parts.append(
            "⚖️ **Compliance Reminder:** Ensure all flagged accounts are reviewed according to regulatory requirements.")

        return "\n".join(response_parts)

    def _provide_banking_fallback(self, question: str) -> str:
        """Provide banking-specific fallback analysis"""
        schema_info = self.sql_analyzer.schema_info or {}

        return f"""
        🏦 **Banking System Status Report**

        📊 **Database Connection:** {'✅ Connected' if self.sql_analyzer.connection else '❌ Disconnected'}
        📋 **Available Tables:** {len(schema_info)} tables found
        🔍 **Query:** "{question}"

        📈 **Available Banking Data:**
        {self._format_available_tables(schema_info)}

        💡 **Suggested Actions:**
        1. Try simpler queries like "show me account data"
        2. Ask for "account types distribution"
        3. Request "dormant accounts summary"
        4. Check "database connection status"

        🛠️ **Technical Status:**
        - Primary Table: {self.sql_analyzer.primary_table or 'Not identified'}
        - AI Model: {'Available' if self.llm_model else 'Not available'}
        """

    def _format_available_tables(self, schema_info: Dict) -> str:
        """Format available tables for display"""
        if not schema_info:
            return "   • No tables available"

        formatted = []
        for table_name, columns in schema_info.items():
            row_count = self.sql_analyzer.get_table_row_count(table_name)
            formatted.append(f"   • {table_name}: {len(columns)} columns, {row_count:,} rows")

        return "\n".join(formatted[:5]) + (
            f"\n   • ... and {len(schema_info) - 5} more tables" if len(schema_info) > 5 else "")


def get_response_and_chart(user_query, current_data, llm_model):
    """
    Main entry point using database connection instead of uploaded data
    """
    try:
        # Initialize banking chatbot with database connectivity
        chatbot = EnhancedBankingChatbot(llm_model)

        # Process query through banking pipeline
        response_text, chart = chatbot.process_banking_query(user_query)

        return response_text, chart

    except Exception as e:
        error_msg = f"❌ **Banking System Error:** {str(e)}"
        st.error(error_msg)

        # Provide system status as fallback
        try:
            db_conn = get_db_connection()
            db_status = "✅ Connected" if db_conn else "❌ Disconnected"

            basic_info = f"""
            🏦 **Banking System Status:**

            🔗 **Database:** {db_status}
            🤖 **AI Model:** {'Available' if llm_model else 'Not Available'}
            📊 **Query:** "{user_query}"

            💡 **System Recovery Options:**
            1. Check database connection settings
            2. Verify API key configuration
            3. Try basic queries like "show tables"
            4. Contact system administrator

            🛠️ **Error Details:** {str(e)[:200]}
            """
            return basic_info, None
        except:
            return error_msg, None


def display_chat_interface(df, llm):
    """
    Enhanced banking compliance chat interface with database integration
    """
    st.header("🏦 Banking Compliance AI Assistant")

    # System status dashboard
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        db_conn = get_db_connection()
        if db_conn:
            st.success("✅ Database")
            st.caption("Azure SQL Connected")
        else:
            st.error("❌ Database")
            st.caption("Connection Failed")

    with col2:
        if llm:
            st.success("✅ AI Model")
            st.caption("LLM Ready")
        else:
            st.error("❌ AI Model")
            st.caption("API Key Needed")

    with col3:
        schema_info = get_db_schema()
        if schema_info:
            st.success(f"✅ Schema ({len(schema_info)} tables)")
            st.caption("Loaded Successfully")
        else:
            st.error("❌ Schema")
            st.caption("Not Available")

    with col4:
        try:
            chatbot = EnhancedBankingChatbot(llm)
            st.success("✅ Chatbot")
            st.caption("Ready for Analysis")
        except:
            st.error("❌ Chatbot")
            st.caption("Initialization Failed")

    # Show system requirements if not met
    if not db_conn or not llm:
        st.error("🚫 **System Requirements Not Met**")

        missing_components = []
        if not db_conn:
            missing_components.append("Database connection (check secrets.toml or environment variables)")
        if not llm:
            missing_components.append("AI model (configure GROQ API key)")

        st.info(f"**Missing:** {', '.join(missing_components)}")

        # Show basic database info if available
        if db_conn:
            with st.expander("📊 Database Schema Preview", expanded=False):
                schema = get_db_schema()
                if schema:
                    for table_name, columns in schema.items():
                        st.write(f"**{table_name}**")
                        col_list = [f"{col[0]} ({col[1]})" for col in columns[:5]]
                        if len(columns) > 5:
                            col_list.append(f"... +{len(columns) - 5} more")
                        st.write(f"   • {', '.join(col_list)}")
        return

    # Banking compliance capabilities
    with st.expander("🏦 Banking Compliance Capabilities", expanded=False):
        st.markdown("""
        **🔥 AI-Powered Banking Analysis:**

        **🎯 Specialized for Banking:**
        - Dormant account identification and analysis
        - Customer communication tracking
        - Compliance flag monitoring
        - Account lifecycle management
        - Regulatory timeline tracking

        **📊 Advanced Analytics:**
        - Account balance distributions
        - Customer activity patterns
        - Dormancy trend analysis
        - Compliance risk assessment
        - Maturity date tracking

        **✨ Example Banking Queries:**
        - *"Show me all dormant accounts"*
        - *"What's the distribution of account types?"*
        - *"Find accounts with balance over $10,000"*
        - *"Show customers with no recent communication"*
        - *"Analyze FTD maturity dates coming up"*

        **🛡️ Compliance Features:**
        - Automatic regulatory flagging
        - Timeline tracking for Article 3 processes
        - Communication requirement monitoring
        - Customer response tracking
        """)

    # Database schema explorer
    with st.expander("📋 Database Schema Explorer", expanded=False):
        schema = get_db_schema()
        if schema:
            selected_table = st.selectbox("Select Table to Explore:", list(schema.keys()))
            if selected_table:
                st.write(f"**Table: {selected_table}**")
                columns = schema[selected_table]

                # Create a formatted display
                col_df = pd.DataFrame(columns, columns=['Column Name', 'Data Type'])
                st.dataframe(col_df, use_container_width=True)

                # Quick query builder
                st.write("**Quick Query Builder:**")
                if st.button(f"Show sample data from {selected_table}"):
                    sample_query = f"Show me sample data from {selected_table}"
                    st.session_state['auto_query'] = sample_query
                    st.rerun()

    # Initialize chat with banking context
    if SESSION_CHAT_MESSAGES not in st.session_state:
        schema = get_db_schema()
        table_count = len(schema) if schema else 0

        initial_message = f"""
        🏦 **Welcome to Banking Compliance AI Assistant!**

        **System Status:**
        ✅ Connected to Azure SQL Database
        ✅ {table_count} banking tables loaded
        ✅ AI compliance analysis ready

        **🎯 I specialize in:**
        - Dormant account analysis and compliance
        - Customer activity and communication tracking  
        - Account balance and type distributions
        - Regulatory timeline monitoring
        - Risk assessment and flagging

        **💡 Try asking:**
        - "Show me dormant accounts that need attention"
        - "What's the breakdown of account types?"
        - "Find high-value accounts with no recent activity"
        - "Analyze customer response rates to bank contact"

        **Ready to help with your banking compliance needs!**
        """
        st.session_state[SESSION_CHAT_MESSAGES] = [{"role": "assistant", "content": initial_message}]

    # Display chat messages
    for i, message in enumerate(st.session_state[SESSION_CHAT_MESSAGES]):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if "chart" in message and message["chart"] is not None:
                try:
                    st.plotly_chart(message["chart"], use_container_width=True, key=f"banking_chart_{i}")
                    st.success("✅ Banking compliance chart generated!")
                except Exception as e:
                    st.error(f"❌ Chart display error: {e}")

    # Banking-specific quick actions
    st.write("**🚀 Banking Quick Actions:**")
    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

    with quick_col1:
        if st.button("🚨 Dormant Accounts", use_container_width=True):
            st.session_state[
                'auto_query'] = "Show me all accounts that are expected to be dormant with their last activity dates"

    with quick_col2:
        if st.button("💰 Account Balances", use_container_width=True):
            st.session_state['auto_query'] = "Show me the distribution of account balances by account type"

    with quick_col3:
        if st.button("📞 Communication Status", use_container_width=True):
            st.session_state['auto_query'] = "Analyze customer communication and response patterns"

    with quick_col4:
        if st.button("📊 Compliance Overview", use_container_width=True):
            st.session_state['auto_query'] = "Give me a comprehensive compliance overview of all accounts"

    # Handle auto queries from quick actions
    prompt = None
    if 'auto_query' in st.session_state:
        prompt = st.session_state['auto_query']
        del st.session_state['auto_query']

    # Chat input
    if not prompt:
        prompt = st.chat_input(
            placeholder="Ask about dormant accounts, compliance status, account analysis..."
        )

    if prompt:
        # Add user message
        st.session_state[SESSION_CHAT_MESSAGES].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Process through banking system
        with st.chat_message("assistant"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                progress_bar.progress(25)
                status_text.text("🏦 Analyzing banking query...")

                progress_bar.progress(50)
                status_text.text("🔍 Querying Azure SQL Database...")

                progress_bar.progress(75)
                status_text.text("📊 Creating compliance visualization...")

                # Process query through banking system
                response_text, chart_obj = get_response_and_chart(prompt, None, llm)

                progress_bar.progress(100)
                status_text.text("✅ Banking analysis complete!")

                # Clear progress
                progress_bar.empty()
                status_text.empty()

                # Display results
                if response_text:
                    st.markdown(response_text)

                if chart_obj is not None:
                    st.plotly_chart(chart_obj, use_container_width=True)
                    st.success("✅ Banking compliance visualization generated!")

                # Add to chat history
                assistant_response = {"role": "assistant", "content": response_text or "Analysis completed."}
                if chart_obj is not None:
                    assistant_response["chart"] = chart_obj
                st.session_state[SESSION_CHAT_MESSAGES].append(assistant_response)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()

                error_msg = f"""
                ❌ **Banking Analysis Error:** {str(e)}

                🔧 **Troubleshooting:**
                1. Check database connection status above
                2. Verify table permissions
                3. Try simpler queries like "show account types"
                4. Use quick action buttons above

                💡 **Alternative Queries:**
                - "What tables are available?"
                - "Show me sample account data"
                - "Check database connection"
                """

                st.error(error_msg)
                st.session_state[SESSION_CHAT_MESSAGES].append({"role": "assistant", "content": error_msg})

    # Enhanced footer with banking context
    st.markdown("---")
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)

    with footer_col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state[SESSION_CHAT_MESSAGES] = []
            st.rerun()

    with footer_col2:
        if st.button("📊 Query History", use_container_width=True):
            try:
                from database.operations import get_recent_sql_history
                history = get_recent_sql_history(5)
                if history is not None and not history.empty:
                    st.dataframe(history, use_container_width=True)
                else:
                    st.info("No query history available")
            except Exception as e:
                st.error(f"Could not load history: {e}")

    with footer_col3:
        if st.button("🔄 Refresh Schema", use_container_width=True):
            get_db_schema.clear()  # Clear cache
            st.success("✅ Schema cache refreshed!")
            st.rerun()

    with footer_col4:
        if st.button("🏦 System Status", use_container_width=True):
            db_status = "✅ Connected" if get_db_connection() else "❌ Disconnected"
            ai_status = "✅ Ready" if llm else "❌ Not Available"
            schema_status = "✅ Loaded" if get_db_schema() else "❌ Not Available"

            st.info(f"""
            **Banking System Status:**
            - Database: {db_status}
            - AI Model: {ai_status}  
            - Schema: {schema_status}
            """)


def render_chatbot(df, llm_model):
    """
    Compatibility wrapper - now uses database instead of uploaded DataFrame
    """
    return display_chat_interface(df, llm_model)