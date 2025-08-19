# --- START OF FILE llm.py ---

import os
import streamlit as st
import re

# Try to import Langchain/Groq
try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
    from langchain_core.output_parsers import StrOutputParser

    LANGCHAIN_AVAILABLE = True
except ImportError:
    st.warning(
        "Langchain/Groq libraries not found. AI features are disabled. Install with: pip install langchain langchain-groq")
    LANGCHAIN_AVAILABLE = False


    # Define dummy classes to avoid NameErrors
    class ChatGroq:
        def __init__(self, **kwargs): pass

        def invoke(self, messages):
            response = AIMessage(content="AI features are disabled because Langchain/Groq libraries are not installed.")
            return response


    # IMPROVEMENT: Simplified dummy PromptTemplate. No need for regex or complex logic.
    class PromptTemplate:
        def __init__(self, template=""):
            self.template = template

        @staticmethod
        def from_template(text):
            return PromptTemplate(template=text)

        def format(self, **kwargs):
            return self.template.format(**kwargs)


    class HumanMessage:
        def __init__(self, content): self.content = content


    class AIMessage:
        def __init__(self, content): self.content = content


    class StrOutputParser:
        def __init__(self): pass

        def invoke(self, input_data):
            return str(getattr(input_data, 'content', input_data))


@st.cache_resource(show_spinner="Loading AI Assistant...")
def load_llm():
    """Loads the Groq LLM using API key from st.secrets or environment variables."""
    if not LANGCHAIN_AVAILABLE:
        st.sidebar.warning("⚠️ AI Assistant Disabled: Langchain/Groq not installed.")
        return None

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("❗️ GROQ API Key not found.")
        st.info("Please set `GROQ_API_KEY` in your `.streamlit/secrets.toml` file or as an environment variable.")
        st.sidebar.error("⚠️ AI Assistant Disabled: API Key missing.")
        return None

    try:
        llm_instance = ChatGroq(temperature=0.2, model_name="llama3-70b-8192", api_key=api_key, request_timeout=120)

        # Test the API key with a simple call
        test_message = [HumanMessage(content="Say 'API key is valid'.")]
        test_response = llm_instance.invoke(test_message)
        if 'api key is valid' in str(test_response.content).lower():
            st.sidebar.success("✅ AI Assistant Loaded!")
            return llm_instance
        else:
            st.warning(
                "Groq API test call returned an unexpected response. Proceeding, but reliability may be affected.")
            st.sidebar.warning("⚠️ AI Assistant Warning: Test failed.")
            return llm_instance
    except Exception as e:
        error_message = str(e).lower()
        if "invalid api key" in error_message or "authentication" in error_message:
            st.error("🚨 Invalid Groq API Key. Please check your key.")
            st.sidebar.error("⚠️ AI Assistant Disabled: Invalid API Key.")
        elif "rate limit" in error_message:
            st.error("🚨 Groq API rate limit exceeded. Please wait and try again.")
            st.sidebar.error("⚠️ AI Assistant Disabled: Rate Limited.")
        else:
            st.error(f"🚨 Failed to initialize or test Groq client: {e}")
            st.sidebar.error("⚠️ AI Assistant Disabled: Initialization Failed.")
        return None


def _invoke_llm(llm, prompt_template, inputs):
    """Helper function to format prompt, invoke LLM, and parse output."""
    if llm is None:
        return get_fallback_response("generic_error")
    try:
        chain = prompt_template | llm | StrOutputParser()
        return chain.invoke(inputs)
    except Exception as e:
        st.error(f"Error during AI invocation: {e}")
        return f"Sorry, an error occurred while generating the AI response: {e}"


def prepare_dataset_info_for_ai(df, analysis_type, sample_size=15):
    """Prepare comprehensive dataset information for AI analysis."""
    sample_df = df.sample(n=min(len(df), sample_size), random_state=42)
    sample_data_csv = sample_df.to_csv(index=False)

    return sample_data_csv, {
        'total_records': len(df),
        'column_names': df.columns.tolist(),
        'analysis_type': analysis_type
    }


def generate_enhanced_ai_insights(llm, df, analysis_type, agent_name):
    """Generate comprehensive AI insights with accurate dataset information."""
    if llm is None:
        return {key: get_fallback_response(key) for key in ['observations', 'trends', 'narration', 'actions']}

    sample_data_csv, dataset_info = prepare_dataset_info_for_ai(df, analysis_type)
    observations = analyze_data_observation_enhanced(llm, sample_data_csv, dataset_info)
    trends = analyze_data_trend_enhanced(llm, sample_data_csv, dataset_info)
    narration = generate_narration(llm, observations, trends)
    actions = generate_actions(llm, observations, trends)
    return {'observations': observations, 'trends': trends, 'narration': narration, 'actions': actions}


def analyze_data_observation_enhanced(llm, data_sample, dataset_info):
    prompt = PromptTemplate.from_template(ENHANCED_OBSERVATION_PROMPT)
    return _invoke_llm(llm, prompt, {"data_sample": data_sample, **dataset_info})


def analyze_data_trend_enhanced(llm, data_sample, dataset_info):
    prompt = PromptTemplate.from_template(ENHANCED_TREND_PROMPT)
    return _invoke_llm(llm, prompt, {"data_sample": data_sample, **dataset_info})


def generate_sql(llm, schema, question):
    prompt = PromptTemplate.from_template(SQL_GENERATION_PROMPT)
    return _invoke_llm(llm, prompt, {"schema": schema, "question": question})


def explain_sql(llm, schema, sql_query):
    prompt = PromptTemplate.from_template(SQL_EXPLANATION_PROMPT)
    return _invoke_llm(llm, prompt, {"schema": schema, "sql_query": sql_query})


def summarize_dormant(llm, analysis_details):
    prompt = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
    return _invoke_llm(llm, prompt, {"analysis_details": analysis_details})


def summarize_compliance(llm, compliance_details):
    prompt = PromptTemplate.from_template(COMPLIANCE_SUMMARY_PROMPT)
    return _invoke_llm(llm, prompt, {"compliance_details": compliance_details})


def analyze_data_observation(llm, data):
    prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
    return _invoke_llm(llm, prompt, {"data": data})


def analyze_data_trend(llm, data):
    prompt = PromptTemplate.from_template(TREND_PROMPT)
    return _invoke_llm(llm, prompt, {"data": data})


def generate_narration(llm, observation, trend):
    prompt = PromptTemplate.from_template(NARRATION_PROMPT)
    return _invoke_llm(llm, prompt, {"observation": observation, "trend": trend})


def generate_actions(llm, observation, trend):
    prompt = PromptTemplate.from_template(ACTION_PROMPT)
    return _invoke_llm(llm, prompt, {"observation": observation, "trend": trend})


# FIX: Consolidated the two get_fallback_response functions into one.
def get_fallback_response(prompt_type, inputs=None):
    """Generate fallback responses when LLM is not available."""
    responses = {
        "dormant_summary": "AI summary not available. Please check API key configuration.",
        "compliance_summary": "AI compliance summary not available. Check API key configuration.",
        "sql_generation": "SELECT * FROM accounts_data LIMIT 10; -- Fallback query.",
        "sql_explanation": "SQL explanation not available. Check API key configuration.",
        "observation": "Data observation not available. AI features are disabled.",
        "trends": "Trend analysis not available. AI features are disabled.",
        "narration": "Executive summary not available. AI features are disabled.",
        "actions": "Recommended actions not available. AI features are disabled.",
        "generic_error": "AI response not available. Please check API key configuration."
    }
    return responses.get(prompt_type, responses["generic_error"])


def clean_sql_query(sql_text):
    """Clean and extract valid SQL from raw text."""
    if not sql_text: return None
    sql_text = re.sub(r"^```sql\s*|\s*```$", "", sql_text, flags=re.MULTILINE).strip()
    match = re.search(r"\bSELECT\b.*", sql_text, re.IGNORECASE | re.DOTALL)
    return match.group(0).strip() if match else sql_text


# --- Prompts ---
DORMANT_SUMMARY_PROMPT = "Act as a Senior Compliance Analyst AI. Based on the following findings, provide a concise executive summary highlighting key risks and trends regarding dormant accounts.\n\n{analysis_details}\n\nExecutive Summary:"
COMPLIANCE_SUMMARY_PROMPT = "Act as a Senior Banking Compliance Officer AI. Based on the following findings, provide a brief executive summary of the most significant compliance risks.\n\n{compliance_details}\n\nCompliance Summary:"
OBSERVATION_PROMPT = "You are a senior bank analyst. Analyze the following sample data and provide key observations about the dataset structure, patterns, and quality.\n\nSample Data (CSV):\n{data}\n\nObservations:"
TREND_PROMPT = "You are a data strategist. Given the following sample data, identify potential trends or significant findings relevant to compliance or risk.\n\nSample Data (CSV):\n{data}\n\nTrends and Findings:"
ENHANCED_OBSERVATION_PROMPT = "You are a senior bank analyst analyzing a dataset of {total_records:,} records. Below is a sample of {total_records} records. Provide key observations about the data structure, patterns, and quality based on this sample, keeping the total dataset size in mind.\n\nSample Data:\n{data_sample}\n\nObservations:"
ENHANCED_TREND_PROMPT = "You are a data strategist analyzing compliance data for '{analysis_type}'. The dataset has {total_records:,} records. Based on the sample below, identify potential trends and risk patterns.\n\nSample Data:\n{data_sample}\n\nTrends and Findings:"
NARRATION_PROMPT = "You are writing a CXO summary. Using the provided observations and trends, craft a concise executive summary (max 3-4 sentences).\n\nObservations:\n{observation}\n\nTrends:\n{trend}\n\nExecutive Summary:"
ACTION_PROMPT = "You are a strategic advisor. Based on the following observations and trends, suggest specific, actionable steps the bank should take.\n\nObservations:\n{observation}\n\nTrends:\n{trend}\n\nRecommended Actions:"
SQL_GENERATION_PROMPT = "You are an expert T-SQL query generator. Given the schema and a question, generate *only* the valid T-SQL query. Do not include any explanations or markdown.\n\nSchema (Azure SQL):\n{schema}\n\nUser question: {question}\n\nT-SQL Query:"
SQL_EXPLANATION_PROMPT = "You are a data analyst. Provide a clear, concise explanation of what the following T-SQL query does.\n\nSchema:\n{schema}\n\nSQL Query:\n```sql\n{sql_query}\n```\n\nExplanation:"
ADVANCED_SQL_GENERATION_PROMPT = "You are an expert T-SQL query generator specializing in advanced features like window functions and CTEs. Given the schema and a question, generate *only* the valid T-SQL query.\n\nSchema (Azure SQL):\n{schema}\n\nUser question: {question}\n\nT-SQL Query:"
ADVANCED_SQL_EXPLANATION_PROMPT = "You are a senior data engineer. Provide a detailed, step-by-step explanation of what the following advanced T-SQL query does.\n\nSchema:\n{schema}\n\nSQL Query:\n```sql\n{sql_query}\n```\n\nDetailed Explanation:"