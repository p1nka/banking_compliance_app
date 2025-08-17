# --- START OF FILE chatbot.py ---

import streamlit as st
import pandas as pd
import re
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import SESSION_CHAT_MESSAGES, SESSION_COLUMN_MAPPING


# FIX: Removed the non-existent import. This function is not used in this file.
# from ui.sqlbot_ui import render_sqlbot

def get_response_and_chart(user_query, current_data, llm_model):
    """
    Processes user query dynamically using LLM to generate a plot or a text answer.
    """
    if llm_model is None:
        return "AI features are disabled. Please check the API key configuration.", None
    if current_data is None or current_data.empty:
        return "⚠️ No data loaded. Please upload and process data first.", None

    try:
        # --- Prepare Context for LLM ---
        cols_info = []
        for col in current_data.columns:
            dtype = str(current_data[col].dtype)
            if pd.api.types.is_numeric_dtype(current_data[col]):
                desc = current_data[col].describe()
                unique_approx = f"Min: {desc['min']:.2f}, Max: {desc['max']:.2f}"
            elif pd.api.types.is_datetime64_any_dtype(current_data[col]):
                unique_approx = f"Date Range: {current_data[col].min():%Y-%m-%d} to {current_data[col].max():%Y-%m-%d}"
            else:
                unique_approx = f"~{current_data[col].nunique()} unique values"
            cols_info.append(f"- `{col}` ({unique_approx})")

        columns_description = "\n".join(cols_info)
        allowed_plots = ['bar', 'pie', 'histogram', 'box', 'scatter']

        interpretation_prompt_text = """
        You are an intelligent assistant interpreting user requests about a dataset.
        Analyze the user's query: "{user_query}"
        Available Data Context: {num_rows} rows, columns: {columns_description}
        Allowed plot types: {allowed_plots_str}

        Task: Determine if the user wants to 'plot' or 'answer'.
        1. If it's a 'plot' request, identify the best plot type and necessary columns.
           Output ONLY a JSON object: {{"action": "plot", "plot_type": "...", "x_column": "...", ...}}
        2. If it's a 'question' or an unclear plot request, output ONLY a JSON object: {{"action": "answer", "query_for_llm": "{user_query}"}}
        Adhere strictly to the JSON format. Use exact column names provided.
        JSON Output:"""

        interpretation_prompt = PromptTemplate.from_template(interpretation_prompt_text)
        prompt_input = {
            "user_query": user_query,
            "num_rows": len(current_data),
            "columns_description": columns_description,
            "allowed_plots_str": ', '.join(allowed_plots)
        }

        interpretation_chain = interpretation_prompt | llm_model | StrOutputParser()
        with st.spinner("Interpreting request..."):
            llm_json_output_str = interpretation_chain.invoke(prompt_input)

        cleaned_json_str = re.sub(r"^```json\s*|\s*```$", "", llm_json_output_str, flags=re.MULTILINE).strip()
        llm_output = json.loads(cleaned_json_str)
        action = llm_output.get("action")

        if action == "plot":
            from .visualizations import generate_plot
            # FIX: Use .get() to avoid KeyError if the session state is not set.
            column_map = st.session_state.get(SESSION_COLUMN_MAPPING, {})
            return generate_plot(llm_output, current_data, column_map)

        elif action == "answer":
            col_context = f"Dataset has {len(current_data)} rows with columns: {', '.join(current_data.columns)}."
            answer_prompt_text = "You are a helpful banking compliance assistant. Answer the user's question based on the provided context.\nContext: {data_context}\nUser Question: {user_question}\nAnswer:"
            answer_prompt = PromptTemplate.from_template(answer_prompt_text)
            answer_chain = answer_prompt | llm_model | StrOutputParser()

            with st.spinner("🤔 Thinking..."):
                response_text = answer_chain.invoke({"data_context": col_context, "user_question": user_query})
            return response_text, None

        else:
            return f"Sorry, I received an unexpected instruction ('{action}'). Please rephrase.", None

    except (json.JSONDecodeError, ValueError) as e:
        return f"Sorry, the AI returned an invalid format. Please try again. (Error: {e})", None
    except Exception as e:
        return f"❌ An unexpected error occurred: {e}", None


def display_chat_interface(df, llm):
    """Display and manage the chat interface."""
    st.header("Banking Compliance Chatbot")

    if df is None or df.empty:
        st.error("No data available. Please upload and process data first.")
        return
    if llm is None:
        st.warning("⚠️ AI Assistant not available. Chat functionality is disabled.")
        return

    st.info("💬 Ask questions or request plots about the loaded data.")

    if SESSION_CHAT_MESSAGES not in st.session_state:
        st.session_state[SESSION_CHAT_MESSAGES] = [{"role": "assistant",
                                                    "content": f"Hi! I'm ready to analyze your data ({len(df)} rows loaded). What would you like to know?"}]

    for message in st.session_state[SESSION_CHAT_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "chart" in message and message["chart"] is not None:
                st.plotly_chart(message["chart"], use_container_width=True)

    if prompt := st.chat_input("e.g., 'Show a bar chart of account types' or 'How many dormant accounts?'"):
        st.session_state[SESSION_CHAT_MESSAGES].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_text, chart_obj = get_response_and_chart(prompt, df, llm)
            st.markdown(response_text)
            if chart_obj:
                st.plotly_chart(chart_obj, use_container_width=True)

            assistant_response = {"role": "assistant", "content": response_text, "chart": chart_obj}
            st.session_state[SESSION_CHAT_MESSAGES].append(assistant_response)