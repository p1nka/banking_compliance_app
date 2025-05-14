import os
import streamlit as st

# Prompts for AI responses
SQL_GENERATION_PROMPT = """
You are a database expert who specializes in translating natural language queries into SQL.

The user will ask questions about their database, and you will generate the correct SQL query to answer their question.

Database Schema:
{schema}

User Question: {question}

Generate a SQL query to answer this question. Be specific and make sure to handle NULL values appropriately. 
Only use tables and columns that appear in the schema provided.
Do not include any explanations, just provide the SQL query itself.

The query should be written for SQL Server/Azure SQL Database.
"""

SQL_EXPLANATION_PROMPT = """
You are a database expert who specializes in explaining SQL queries in simple terms.

Database Schema:
{schema}

SQL Query:
{sql_query}

Please explain this SQL query in clear, simple language that a non-technical person could understand. 
Include the following in your explanation:
1. What data is being retrieved or modified
2. What conditions/filters are being applied
3. How the results are organized or sorted
4. Any potential performance considerations

Your explanation should be concise yet comprehensive.
"""

DORMANT_SUMMARY_PROMPT = """
You are a financial compliance expert summarizing the results of a dormant account analysis.

Here is the raw analysis data:
{analysis_details}

Please provide a clear, executive-level summary of these findings. Your summary should:
1. Highlight the key numbers and their significance
2. Identify any patterns or trends
3. Explain the implications for banking compliance
4. Use professional but accessible language

Make your summary comprehensive but concise, suitable for a busy banking executive.
"""

COMPLIANCE_SUMMARY_PROMPT = """
You are a financial compliance expert summarizing the results of a banking compliance analysis.

Here is the raw compliance data:
{compliance_details}

Please provide a clear, executive-level summary of these findings. Your summary should:
1. Highlight the key compliance issues and their significance
2. Identify any regulatory risks
3. Explain the implications for banking operations
4. Suggest high-level next steps

Make your summary comprehensive but concise, suitable for a busy banking executive.
"""

OBSERVATION_PROMPT = """
Analyze the following dormant account data and provide key observations:

```
{data}
```

Provide 3-5 specific observations about the data. Focus on patterns, anomalies, or notable characteristics 
that would be relevant to a banking compliance officer. Be specific and quantitative when possible.
"""

TREND_PROMPT = """
Analyze the following dormant account data and identify trends:

```
{data}
```

Identify 2-4 significant trends or patterns in this data. Focus on changes over time, 
relationships between variables, or clusters/groupings that would be relevant for compliance risk assessment.
Be specific and insightful in your analysis.
"""

NARRATION_PROMPT = """
Based on the observations and trends identified in dormant account analysis, 
create a concise executive summary:

Observations:
{observation}

Trends:
{trend}

Write a concise, clear summary that synthesizes the above information into a coherent narrative 
about the dormant accounts situation. Focus on the big picture and implications for compliance management.
This should be suitable for a banking executive who needs to understand the situation quickly.
"""

ACTION_PROMPT = """
Based on the dormant account analysis, recommend specific actions:

Observations:
{observation}

Trends:
{trend}

Provide 3-5 specific, actionable recommendations for the bank's compliance team. 
Each recommendation should:
1. Directly address an issue identified in the analysis
2. Be specific and concrete
3. Include a clear rationale
4. Prioritize regulatory compliance and risk mitigation

Format each recommendation as a clear action item.
"""


def get_llm():
    """
    Get a language model instance for generating AI responses.
    Returns None if LangChain is not installed or no API key is found.
    """
    # First check if LangChain is installed
    try:
        import langchain
    except ImportError:
        st.sidebar.warning("⚠️ LangChain not installed. AI features will be unavailable.")
        return None

    # Then try to get Groq module
    try:
        from langchain.llms import Groq
    except ImportError:
        try:
            # Try newer version of LangChain
            from langchain_groq import ChatGroq
            st.sidebar.info("Using ChatGroq from langchain_groq")
        except ImportError:
            st.sidebar.warning("⚠️ Groq integration not installed. Run: pip install langchain-groq")
            return None

    # Try to get API key
    groq_api_key = None

    # Check Streamlit secrets first
    if hasattr(st, 'secrets') and 'GROQ_API_KEY' in st.secrets:
        groq_api_key = st.secrets['GROQ_API_KEY']

    # Fall back to environment variables
    if not groq_api_key:
        groq_api_key = os.getenv('GROQ_API_KEY')

    # If no API key is available, return None
    if not groq_api_key:
        st.sidebar.warning("⚠️ GROQ API key not found in secrets or environment variables.")
        st.sidebar.info(
            "To enable AI features, add GROQ_API_KEY to your .streamlit/secrets.toml or as an environment variable.")
        return None

    # Create and return the LLM
    try:
        # Try with langchain.llms (older version)
        try:
            from langchain.llms import Groq
            llm = Groq(
                api_key=groq_api_key,
                model_name="llama3-70b-8192",
                max_tokens=4000,
                temperature=0.2,
            )
            st.sidebar.success("✅ Connected to Groq API (llama3-70b)")
            return llm
        except (ImportError, AttributeError):
            # Try with newer langchain_groq
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    api_key="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",
                    model_name="llama3-70b-8192",
                    max_tokens=4000,
                    temperature=0.2,
                )
                st.sidebar.success("✅ Connected to Groq API (llama3-70b) using ChatGroq")
                return llm
            except ImportError:
                st.sidebar.error("Failed to create Groq LLM. Please install with: pip install langchain-groq")
                return None
    except Exception as e:
        st.sidebar.error(f"Error initializing LLM: {e}")
        return None


def get_fallback_response(response_type):
    """
    Get a fallback response when the LLM is not available or fails.

    Args:
        response_type: The type of response needed

    Returns:
        A fallback response string
    """
    fallback_responses = {
        "sql_generation": "SELECT TOP 10 * FROM accounts_data WHERE Expected_Account_Dormant = 'Yes'",
        "sql_explanation": "This query fetches basic information about dormant accounts from the database.",
        "dormant_summary": "Analysis shows multiple dormant accounts requiring review according to compliance guidelines.",
        "compliance_summary": "Several accounts require attention for compliance with regulatory requirements.",
        "observation": "The data shows multiple dormant accounts with varying inactivity periods.",
        "trend": "There appears to be an increasing trend in dormant accounts over time.",
        "narration": "The bank has several dormant accounts that require attention to maintain regulatory compliance.",
        "action": "Review identified dormant accounts and initiate contact with account holders following regulated procedures."
    }

    return fallback_responses.get(response_type, "AI response not available.")