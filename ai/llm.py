import os
import streamlit as st
import re  # Make sure this is included for regex in the dummy classes

# Try to import Langchain/Groq
try:
    # These are the core components needed for interacting with the model
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser
    # Add any other required Langchain components here if needed later

    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.warning(
        "Langchain/Groq libraries not found. AI features will be disabled. Install with: pip install langchain langchain-groq"
    )
    LANGCHAIN_AVAILABLE = False

    # Define dummy classes/functions if Langchain is not available to avoid NameErrors later
    # These dummies mimic the basic interface needed by the rest of the app.
    class ChatGroq:
        def __init__(self, **kwargs):
            # print("Dummy ChatGroq initialized") # Optional debug
            pass

        def invoke(self, messages):
            # Simulate AIMessage structure with content attribute
            dummy_response = lambda: None  # Create a simple object
            dummy_response.content = "AI features are disabled because Langchain/Groq libraries are not installed."
            # print(f"Dummy ChatGroq invoked, returning: {dummy_response.content}") # Optional debug
            return dummy_response

    class PromptTemplate:
        def __init__(self):
            # print("Dummy PromptTemplate initialized") # Optional debug
            pass

        @staticmethod
        def from_template(text):
            # print(f"Dummy PromptTemplate.from_template('{text[:50]}...')") # Optional debug
            # Store template text for potential inspection later if needed
            # Basic extraction of variables (might not be perfectly robust for complex templates)
            input_variables = re.findall(r"\{(\w+)\}", text)

            # Return a dummy object that can be invoked (returning formatted text)
            class DummyPrompt:
                def __init__(self, template, variables):
                    self.template = template
                    self.input_variables = variables
                    # print(f"DummyPrompt created with template and variables: {variables}") # Optional debug

                def format(self, **kwargs):
                    # print(f"DummyPrompt format called with kwargs: {kwargs}") # Optional debug
                    formatted_text = self.template
                    for key, value in kwargs.items():
                        placeholder = f"{{{key}}}"
                        # Ensure value is treated as a string and handle None
                        value_str = str(value) if value is not None else ""
                        # Simple replacement - doesn't handle nested braces or complex formatting
                        # Escape curly braces in the value to prevent format errors if value contains them
                        escaped_value = value_str.replace("{", "{{").replace("}", "}}")
                        formatted_text = formatted_text.replace(placeholder, escaped_value)
                    # print(f"DummyPrompt formatted text: {formatted_text[:50]}...") # Optional debug
                    return formatted_text

            return DummyPrompt(text, input_variables)

    class HumanMessage:
        def __init__(self, content):
            # print(f"Dummy HumanMessage created with content: {content[:50]}...") # Optional debug
            self.content = content  # Basic init for dummy

    class AIMessage:
        def __init__(self, content):
            # print(f"Dummy AIMessage created with content: {content[:50]}...") # Optional debug
            self.content = content  # Basic init for dummy

    class StrOutputParser:
        def __init__(self):
            # print("Dummy StrOutputParser initialized") # Optional debug
            pass  # Dummy init

        def invoke(self, input_data):
            # print(f"Dummy StrOutputParser invoked with data type: {type(input_data)}") # Optional debug
            # Assuming input_data is the AIMessage object or similar
            if hasattr(input_data, 'content'):
                return str(input_data.content)
            return str(input_data)  # Fallback


@st.cache_resource(show_spinner="Loading AI Assistant...")
def load_llm():
    """Loads the Groq LLM using API key from st.secrets or environment variables."""
    if not LANGCHAIN_AVAILABLE:
        st.warning("Langchain/Groq not available. AI features will be disabled.")
        st.sidebar.warning("⚠️ AI Assistant Disabled: Langchain/Groq not installed.")
        return None

    api_key = None
    secrets_available = hasattr(st, 'secrets')

    # Prioritize Streamlit secrets
    if secrets_available:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception as e:
            st.error(f"Error accessing GROQ_API_KEY secrets: {e}. Trying environment variable.")
            st.sidebar.error("⚠️ AI Assistant Disabled: Secret access error.")

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
        st.sidebar.error("⚠️ AI Assistant Disabled: API Key missing.")
        return None

    try:
        # Use a smaller model if 70b is too slow or costly, e.g., "llama3-8b-8192"
        # Configure request_timeout for potentially long-running queries or summaries
        llm_instance = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=api_key, request_timeout=120)

        # Test the API key with a simple call
        try:
            # Use a trivial prompt to test connectivity and key validity
            # Adding system instruction for more predictable "API key is valid" response
            test_message = [
                HumanMessage(content="Say only 'API key is valid'. Do not include any other text.")
                ]
            # The dummy invoke returns an object with .content, so this structure works for both
            test_response = llm_instance.invoke(test_message)
            # Check the response content, allowing for slight variations in casing or whitespace
            if test_response is not None and hasattr(test_response, 'content') and 'api key is valid' in str(test_response.content).strip().lower():
                 st.sidebar.success("✅ AI Assistant Loaded!")
                 return llm_instance
            else:
                 # This might catch cases where the API call succeeds but the model doesn't follow instructions
                 st.warning("Groq API test call did not return expected response. API key might be rate-limited, model issue, or a non-critical error.")
                 st.sidebar.warning("⚠️ AI Assistant Load Warning: Test failed.")
                 # Decide whether to return the instance anyway or None. Returning the instance might still allow some functionality. Let's return it.
                 st.info("Attempting to proceed with AI features, but reliability might be affected.")
                 return llm_instance # Return instance despite test warning
        except Exception as test_e:
            # Catch specific Groq errors if possible, or general Exception
            error_message = str(test_e).lower()
            if "invalid_api_key" in error_message or "invalid api key" in error_message or "authentication" in error_message:
                st.error("🚨 Invalid Groq API Key. Please check your API key and try again.")
                st.sidebar.error("⚠️ AI Assistant Disabled: Invalid API Key.")
            elif "rate limit" in error_message:
                 st.error("🚨 Groq API rate limit exceeded. Please wait a moment and try again.")
                 st.sidebar.error("⚠️ AI Assistant Disabled: Rate Limited.")
            elif "network" in error_message or "timeout" in error_message or "connection" in error_message:
                 st.error(f"🚨 Network or Timeout error contacting Groq API: {test_e}")
                 st.sidebar.error("⚠️ AI Assistant Disabled: Network Error.")
            else:
                st.error(f"🚨 Error during Groq test call: {test_e}")
                st.sidebar.error("⚠️ AI Assistant Disabled: Test Error.")
            return None # Return None on test failure

    except Exception as e:
        # Catch errors during client initialization (e.g., invalid model name)
        st.error(f"🚨 Failed to initialize Groq client: {e}")
        st.info("Please verify your GROQ_API_KEY value and model name.")
        st.sidebar.error("⚠️ AI Assistant Disabled: Initialization Failed.")
        return None

# --- Helper Function to Invoke LLM ---
def _invoke_llm(llm, prompt_template, inputs):
    """
    Helper function to format prompt, invoke LLM, and parse output.
    Handles potential errors during LLM invocation.

    Args:
        llm: The loaded LLM instance (expected to be not None).
        prompt_template (PromptTemplate): The Langchain PromptTemplate object.
        inputs (dict): Dictionary of input variables for the prompt.

    Returns:
        str: The generated text from the LLM, or an error message string.
    """
    if llm is None:
         # This case should ideally be handled by the caller, but good practice
        return "Error: LLM is not loaded."

    try:
        # 1. Format the prompt
        formatted_prompt = prompt_template.format(**inputs)

        # 2. Create message(s) - Simple case is just a HumanMessage
        messages = [HumanMessage(content=formatted_prompt)]

        # 3. Invoke the LLM
        ai_message = llm.invoke(messages)

        # 4. Parse the output
        parser = StrOutputParser()
        output_text = parser.invoke(ai_message)

        return output_text

    except Exception as e:
        # Catch errors during the LLM call itself (e.g., content moderation, API error)
        st.error(f"Error during AI invocation: {e}")
        return f"Sorry, an error occurred while generating the AI response: {e}"


# --- Task-Specific Functions using the LLM ---

# These functions assume they are called *only* when llm is not None.
# The calling code must check for None.

def generate_sql(llm, schema, question):
    """Generates a SQL query using the LLM."""
    prompt = PromptTemplate.from_template(SQL_GENERATION_PROMPT)
    inputs = {"schema": schema, "question": question}
    return _invoke_llm(llm, prompt, inputs)

def explain_sql(llm, schema, sql_query):
    """Explains a SQL query using the LLM."""
    prompt = PromptTemplate.from_template(SQL_EXPLANATION_PROMPT)
    inputs = {"schema": schema, "sql_query": sql_query}
    return _invoke_llm(llm, prompt, inputs)

def summarize_dormant(llm, analysis_details):
    """Generates a summary of dormant account analysis using the LLM."""
    prompt = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
    inputs = {"analysis_details": analysis_details}
    return _invoke_llm(llm, prompt, inputs)

def summarize_compliance(llm, compliance_details):
    """Generates a summary of compliance findings using the LLM."""
    prompt = PromptTemplate.from_template(COMPLIANCE_SUMMARY_PROMPT)
    inputs = {"compliance_details": compliance_details}
    return _invoke_llm(llm, prompt, inputs)

def analyze_data_observation(llm, data):
    """Generates observations based on sample data using the LLM."""
    prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
    inputs = {"data": data}
    return _invoke_llm(llm, prompt, inputs)

def analyze_data_trend(llm, data):
    """Identifies trends based on sample data using the LLM."""
    prompt = PromptTemplate.from_template(TREND_PROMPT)
    inputs = {"data": data}
    return _invoke_llm(llm, prompt, inputs)

def generate_narration(llm, observation, trend):
    """Generates an executive summary based on observations and trends using the LLM."""
    prompt = PromptTemplate.from_template(NARRATION_PROMPT)
    inputs = {"observation": observation, "trend": trend}
    return _invoke_llm(llm, prompt, inputs)

def generate_actions(llm, observation, trend):
    """Suggests recommended actions based on observations and trends using the LLM."""
    prompt = PromptTemplate.from_template(ACTION_PROMPT)
    inputs = {"observation": observation, "trend": trend}
    return _invoke_llm(llm, prompt, inputs)


# --- Common prompts for different use cases ---
# These remain the same as they define the instructions for the LLM
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

OBSERVATION_PROMPT = """You are a senior bank analyst. Analyze the following sample data from identified dormant/inactive accounts and provide only key observations about the dataset structure, common values, or patterns. Format the output as bullet points if appropriate.

Sample Data (CSV):
{data}

Observations:"""

TREND_PROMPT = """You are a data strategist. Given the following sample data from a banking compliance analysis, identify potential trends or significant findings relevant to compliance or risk. Format the output as bullet points if appropriate.

Sample Data (CSV):
{data}

Trends and Findings:"""

NARRATION_PROMPT = """You are writing a CXO summary based on compliance analysis. Using the provided observations and trends, craft a concise executive summary (max 3-4 sentences) suitable for a busy executive.

Observations:
{observation}

Trends:
{trend}

Executive Summary:"""

ACTION_PROMPT = """You are a strategic advisor to a bank's compliance department. Based on the following observations and trends, suggest specific, actionable steps the bank should take to address the identified issues. Format the output as actionable steps, potentially using bullet points.

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

SQL_EXPLANATION_PROMPT = """You are a data analyst explaining an SQL query. Provide a clear, concise explanation of what the following T-SQL query does, referencing the provided database schema. Format the explanation clearly.

Database Schema:
{schema}

SQL Query:
```sql
{sql_query}
Use code with caution.
Python
Explanation:"""
