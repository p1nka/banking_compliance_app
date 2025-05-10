import os
import streamlit as st
import re  # Make sure this is included for regex in the dummy classes

# Try to import Langchain/Groq
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.warning(
        "Langchain/Groq libraries not found. AI features will be disabled. Install with: pip install langchain langchain-groq"
    )
    LANGCHAIN_AVAILABLE = False


    # Define dummy classes/functions if Langchain is not available to avoid NameErrors later
    class ChatGroq:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            # Simulate AIMessage structure with content attribute
            dummy_response = lambda: None
            dummy_response.content = "AI features are disabled. Please install required packages."
            return dummy_response


    class PromptTemplate:
        def __init__(self):
            pass

        @staticmethod
        def from_template(text):
            # Store template text for potential inspection later if needed
            prompt_template = lambda: None  # Simple object to hold attributes
            prompt_template.template = text
            # Basic extraction of variables (might not be perfectly robust for complex templates)
            prompt_template.input_variables = re.findall(r"\{(\w+)\}", text)

            # Return a dummy object that can be invoked (returning formatted text)
            class DummyPrompt:
                def __init__(self, template, variables):
                    self.template = template
                    self.input_variables = variables

                def format(self, **kwargs):
                    formatted_text = self.template
                    for key, value in kwargs.items():
                        # Escape braces in values to prevent accidental formatting
                        escaped_value = str(value).replace("{", "{{").replace("}", "}}")
                        formatted_text = formatted_text.replace(f"{{{key}}}", escaped_value)
                    return formatted_text

            return DummyPrompt(text, prompt_template.input_variables)


    class HumanMessage:
        def __init__(self, content): self.content = content  # Basic init for dummy


    class AIMessage:
        def __init__(self, content): self.content = content  # Basic init for dummy


    class StrOutputParser:
        def __init__(self): pass  # Dummy init

        def invoke(self, input_data):
            # Assuming input_data is the AIMessage object or similar
            if hasattr(input_data, 'content'):
                return str(input_data.content)
            return str(input_data)  # Fallback


@st.cache_resource(show_spinner="Loading AI Assistant...")
def load_llm():
    """Loads the Groq LLM using API key from st.secrets or environment variables."""
    if not LANGCHAIN_AVAILABLE:
        st.warning("Langchain/Groq not available. AI features will be disabled.")
        return None

    api_key = None
    secrets_available = hasattr(st, 'secrets')

    # Prioritize Streamlit secrets
    if secrets_available:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception as e:
            st.error(f"Error accessing GROQ_API_KEY secrets: {e}. Trying environment variable.")

    # Fallback to Environment Variable
    if api_key is None:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❗️ GROQ API Key not found.")
        st.info(
            "To use the AI features, please ensure:\n"
            "1. The file `.streamlit/secrets.toml` exists and contains `GROQ_API_KEY = \"YOUR_ACTUAL_GROQ_API_KEY\"` OR\n"
            "2. A `GROQ_API_KEY` environment variable is set.\n"
            "3. You have restarted the Streamlit app after setting."
        )
        return None

    try:
        # Use a smaller model if 70b is too slow or costly, e.g., "llama3-8b-8192"
        llm_instance = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=api_key, request_timeout=120)

        # Test the API key with a simple call
        try:
            # Use a trivial prompt to test connectivity
            test_message = [HumanMessage(content="Say 'API key is valid'")]
            _ = llm_instance.invoke(test_message)
            st.sidebar.success("✅ AI Assistant Loaded!")
            return llm_instance
        except Exception as test_e:
            if "invalid_api_key" in str(test_e).lower() or "invalid api key" in str(test_e).lower():
                st.error("🚨 Invalid Groq API Key. Please check your API key and try again.")
            else:
                st.error(f"🚨 Error testing Groq client: {test_e}")
            return None
    except Exception as e:
        st.error(f"🚨 Failed to initialize Groq client: {e}")
        st.info("Please verify your GROQ_API_KEY value and ensure you have internet connectivity.")
        return None


# Function to create a fallback response when LLM is not available
def get_fallback_response(prompt_type, inputs=None):
    """
    Generate fallback responses when LLM is not available.

    Args:
        prompt_type (str): The type of prompt (summary, explanation, etc.)
        inputs (dict, optional): Input parameters to customize the response

    Returns:
        str: A generic fallback response
    """
    if prompt_type == "dormant_summary":
        return "AI summary not available. Please check your Groq API key configuration to enable AI features."

    elif prompt_type == "compliance_summary":
        return "AI compliance summary not available. Please check your Groq API key configuration to enable AI features."

    elif prompt_type == "sql_generation":
        return "SELECT * FROM accounts_data LIMIT 10; -- AI-generated SQL not available. This is a fallback query."

    elif prompt_type == "sql_explanation":
        return "SQL query explanation not available. Please check your Groq API key configuration to enable AI features."

    elif prompt_type == "observation":
        return "Data observation not available. AI features are disabled due to API configuration issues."

    elif prompt_type == "trend":
        return "Trend analysis not available. AI features are disabled due to API configuration issues."

    elif prompt_type == "narration":
        return "Executive summary not available. AI features are disabled due to API configuration issues."

    elif prompt_type == "action":
        return "Recommended actions not available. AI features are disabled due to API configuration issues."

    else:
        return "AI response not available. Please check your Groq API key configuration to enable AI features."


# Common prompts for different use cases
DORMANT_SUMMARY_PROMPT = """Act as a Senior Compliance Analyst AI. You have analyzed a dataset of banking accounts.
Below is a numerical summary of accounts identified by specific dormant/inactivity criteria:

{analysis_details}

Based on these findings, provide a concise executive summary highlighting key risks, trends, and observations regarding dormant accounts. Keep it professional and focused on compliance implications.

Executive Summary:"""

COMPLIANCE_SUMMARY_PROMPT = """Act as a Senior Banking Compliance Officer AI. You have reviewed the output of several compliance agents.
Here are the counts and descriptions of accounts identified by each compliance check:

{compliance_details}

Provide a brief executive summary highlighting the most significant compliance risks and areas requiring immediate attention based on these findings. Focus on actionable insights for the compliance team.

Compliance Summary:"""

OBSERVATION_PROMPT = """You are a senior bank analyst. Analyze the following sample data from identified dormant/inactive accounts and provide only key observations about the dataset structure, common values, or patterns.

Sample Data (CSV):
{data}

Observations:"""

TREND_PROMPT = """You are a data strategist. Given the following sample data from a banking compliance analysis, identify potential trends or significant findings relevant to compliance or risk.

Sample Data (CSV):
{data}

Trends and Findings:"""

NARRATION_PROMPT = """You are writing a CXO summary based on compliance analysis. Using the provided observations and trends, craft a concise executive summary (max 3-4 sentences) suitable for a busy executive.

Observations:
{observation}

Trends:
{trend}

Executive Summary:"""

ACTION_PROMPT = """You are a strategic advisor to a bank's compliance department. Based on the following observations and trends, suggest specific, actionable steps the bank should take to address the identified issues.

Observations:
{observation}

Trends:
{trend}

Recommended Actions:"""

SQL_GENERATION_PROMPT = """You are an expert Azure SQL query generator. Given the database schema and a user question in natural language, generate *only* the valid T-SQL query that answers the question.
Adhere strictly to the schema provided. Only use tables and columns exactly as they are named in the schema.
The database is Azure SQL Server. Ensure the query syntax is correct for T-SQL.
Prioritize using the 'accounts_data' table for general account questions. Use 'dormant_flags' or 'sql_query_history' if the question is specifically about those logs.
The user expects a query to *retrieve* data. Generate *only* SELECT statements. Do NOT generate INSERT, UPDATE, DELETE, CREATE, ALTER, or DROP statements, or any other SQL commands.
Do NOT include any explanations, greetings, or markdown code block formatting (```sql```) around the query. Just output the plain SQL query text.
If the question cannot be answered using *only* the provided schema and *only* a SELECT query, output a polite message stating that you cannot answer that query based on the schema.

Database Schema (Azure SQL):
{schema}

User question: {question}

T-SQL Query:"""

SQL_EXPLANATION_PROMPT = """You are a data analyst explaining an SQL query. Provide a clear, concise explanation of what the following T-SQL query does, referencing the provided database schema.

Database Schema:
{schema}

SQL Query:
```sql
{sql_query}
```

Explanation:"""