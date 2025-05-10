import streamlit as st
import pandas as pd
import re
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import SESSION_CHAT_MESSAGES, SESSION_COLUMN_MAPPING
from ai.llm import get_fallback_response


def get_response_and_chart(user_query, current_data, llm_model):
    """
    Processes user query dynamically using LLM. Determines if it's a plot request
    or a question, generates the plot or answers accordingly.
    Handles JSON parsing and potential errors.
    """
    chart = None
    response_text = "Sorry, something went wrong processing your request."  # Default error message

    if not llm_model:
        return "⚠️ AI Assistant not available (check API key or install Langchain). Cannot process dynamic requests.", None

    if current_data is None or current_data.empty:
        return "⚠️ No data loaded. Please upload and process data first.", None

    # --- Prepare Context for LLM ---
    try:
        cols_info = []
        for col in current_data.columns:
            dtype = str(current_data[col].dtype)
            # Provide more specific info based on dtype
            if pd.api.types.is_numeric_dtype(current_data[col]):
                if current_data[col].notna().any():
                    desc = current_data[col].describe()
                    unique_approx = f"Min: {desc['min']:.2f}, Max: {desc['max']:.2f}, Mean: {desc['mean']:.2f}"
                else:
                    unique_approx = "No numeric data"
            elif pd.api.types.is_datetime64_any_dtype(current_data[col]):
                if current_data[col].notna().any():
                    unique_approx = f"Date Range: {current_data[col].min().strftime('%Y-%m-%d')} to {current_data[col].max().strftime('%Y-%m-%d')}"
                else:
                    unique_approx = "No valid dates"
            elif pd.api.types.is_string_dtype(current_data[col]):
                unique_count = current_data[col].nunique()
                unique_approx = f"~{unique_count} unique values"
                if unique_count > 0 and unique_count <= 20:  # List categories if few
                    unique_vals = current_data[col].dropna().unique()
                    unique_approx += f" (Values: {', '.join(map(str, unique_vals))})"
                elif unique_count > 20:
                    unique_approx += f" (Top 3 e.g., {', '.join(map(str, current_data[col].value_counts().nlargest(3).index))})"
            else:
                unique_approx = f"Type: {dtype}"  # Generic fallback

            cols_info.append(f"- `{col}` ({unique_approx})")

        columns_description = "\n".join(cols_info)
        num_rows = len(current_data)
        # Allowed plot types relevant to common business data analysis
        allowed_plots = ['bar', 'pie', 'histogram', 'box', 'scatter']

        interpretation_prompt_text = """
You are an intelligent assistant interpreting user requests about a banking compliance dataset.
Analyze the user's query: "{user_query}"
Available Data Context:
- Number of rows: {num_rows}
- Available standardized columns and their details:
{columns_description}
- Allowed plot types (from plotly.express): {allowed_plots_str}
- Pie charts require a categorical column with few (<25) unique values.
- Histograms and Box plots are best for numeric or date columns.
- Bar charts are good for counts by category.
- Scatter plots need two numeric/date columns.

Task:
1. Determine if the user query is primarily a request to **plot** data or a **question** to be answered.
2. If it's a **plotting request** AND seems feasible with the available columns and allowed plot types:
   - Identify the most appropriate plot type from the allowed list ({allowed_plots_str}).
   - Identify the necessary column(s) for that plot type using the **exact standardized column names** provided above.
     - For 'bar': `x_column` (category), optional `color_column`.
     - For 'pie': `names_column` (categorical, few unique values). `values_column` is implicitly the count.
     - For 'histogram': `x_column` (numeric/date), optional `color_column`.
     - For 'box': `y_column` (numeric), optional `x_column` (category), optional `color_column`.
     - For 'scatter': `x_column` (numeric/date), `y_column` (numeric/date), optional `color_column`.
   - Generate a concise, suitable title for the plot based on the query and columns used.
   - Output **ONLY** a valid JSON object with the following structure. Use `null` (JSON null) for unused keys.
     ```json
     {{
       "action": "plot",
       "plot_type": "chosen_plot_type",
       "x_column": "Standardized_X_Column_Name_or_null",
       "y_column": "Standardized_Y_Column_Name_or_null",
       "color_column": "Standardized_Color_Column_Name_or_null",
       "names_column": "Standardized_Pie_Names_Column_or_null",
       "title": "Suggested Plot Title"
     }}
     ```
3. If the query is a **question**, a request for analysis/summary, or an infeasible/unclear plot request:
   - Output **ONLY** a valid JSON object with the structure:
     ```json
     {{
       "action": "answer",
       "query_for_llm": "{user_query}"
     }}
     ```
Constraints:
- Adhere strictly to the JSON format specified for each action.
- Only use standardized column names listed in the context.
- Only use plot types from the allowed list: {allowed_plots_str}.
- If a plot request is ambiguous or infeasible based on column types/values, default to the "answer" action.
- The JSON output must be the *only* text generated. Do NOT add any explanations, greetings, or markdown formatting around the JSON.
JSON Output:
"""
        interpretation_prompt = PromptTemplate.from_template(interpretation_prompt_text)
        prompt_input = {
            "user_query": user_query,
            "num_rows": num_rows,
            "columns_description": columns_description,
            "allowed_plots_str": ', '.join(allowed_plots)
        }

        try:
            # Create the LLM chain for interpretation
            interpretation_chain = interpretation_prompt | llm_model | StrOutputParser()

            with st.spinner("Interpreting request..."):
                llm_json_output_str = interpretation_chain.invoke(prompt_input)

            # Process the LLM response
            try:
                # Clean the output - sometimes LLMs wrap JSON in ```json ... ```
                cleaned_json_str = re.sub(r"^```json\s*|\s*```$", "", llm_json_output_str, flags=re.MULTILINE).strip()
                if not cleaned_json_str:
                    raise ValueError("LLM returned an empty response after cleaning.")

                llm_output = json.loads(cleaned_json_str)
                action = llm_output.get("action")

                if action == "plot":
                    from .visualizations import generate_plot
                    try:
                        chart, response_text = generate_plot(llm_output, current_data,
                                                             st.session_state[SESSION_COLUMN_MAPPING])
                    except Exception as plot_e:
                        st.error(f"Error generating plot: {plot_e}")
                        response_text = f"I understood you want a visualization, but encountered an error: {plot_e}. Please try a different type of chart or query."
                        chart = None

                elif action == "answer":
                    query_to_answer = llm_output.get("query_for_llm", user_query)

                    # Create context about the dataset
                    col_context = f"Dataset has {len(current_data)} rows. Standardized columns available: {', '.join(current_data.columns)}. "

                    if SESSION_COLUMN_MAPPING in st.session_state and st.session_state[SESSION_COLUMN_MAPPING]:
                        # Show original names in context for better understanding by LLM
                        original_names_map = {
                            st.session_state[SESSION_COLUMN_MAPPING][std_col]: std_col
                            for std_col in current_data.columns
                            if std_col in st.session_state[SESSION_COLUMN_MAPPING]
                        }
                        if original_names_map:
                            col_context += f"Original column names and their standardized versions: {'; '.join([f'{orig} (std: {std})' for orig, std in original_names_map.items()])}. "

                    answer_prompt_text = """
You are a helpful banking compliance assistant. Answer the user's question based on the provided context about the loaded dataset.
Be concise and directly address the question. If the question asks for specific numbers or analysis not directly available from simple column value counts, state that you can only provide insights based on the available columns and cannot perform complex calculations on the entire dataset interactively.

Context about the dataset:
{data_context}

User Question: {user_question}

Answer:"""
                    answer_prompt = PromptTemplate.from_template(answer_prompt_text)

                    try:
                        answer_chain = answer_prompt | llm_model | StrOutputParser()
                        with st.spinner("🤔 Thinking..."):
                            ai_response_content = answer_chain.invoke({
                                "data_context": col_context,
                                "user_question": query_to_answer
                            })
                        response_text = ai_response_content if ai_response_content else "Sorry, I couldn't formulate an answer."
                    except Exception as answer_e:
                        st.error(f"Error generating answer: {answer_e}")
                        response_text = f"I understood your question but encountered an error while answering: {answer_e}. Here's what I know about the data: {col_context}"

                    chart = None  # Ensure chart is None for answers

                else:
                    response_text = f"Sorry, I received an unexpected instruction ('{action}') from the AI interpreter. Please rephrase your request."
                    chart = None  # Ensure chart is None

            except (json.JSONDecodeError, ValueError) as e:
                response_text = f"Sorry, there was an error processing the AI's response ({e}). It might not have returned the expected format. Please try rephrasing."
                chart = None  # Ensure chart is None
            except Exception as e:
                response_text = f"❌ An unexpected error occurred during interpretation: {e}"
                chart = None  # Ensure chart is None

        except Exception as llm_e:
            response_text = f"I encountered an error while processing your request with the AI model: {llm_e}. This might be due to an issue with the API key or connectivity."
            chart = None

    except Exception as e:
        # Catch errors during prompt preparation or initial LLM invocation
        response_text = f"❌ Failed to process your request due to an internal error: {e}"
        chart = None  # Ensure chart is None

    return response_text, chart


def display_chat_interface(df, llm):
    """
    Display and manage the chat interface.

    Args:
        df: DataFrame containing the data to analyze
        llm: LLM model for generating responses
    """
    st.header("Banking Compliance Chatbot")

    if df is None or df.empty:
        st.error("No data available. Please upload and process data first.")
        return

    if llm is None:
        st.warning("⚠️ AI Assistant not available. Chat functionality will be limited.")
        st.info(
            "To enable full chatbot functionality, please configure your GROQ API key in secrets.toml or as an environment variable.")

        # Still display a simple input but with limited functionality
        simple_query = st.text_input("Enter a simple query about the data (basic features only):")
        if simple_query:
            st.markdown(f"### Query: {simple_query}")

            # Basic dataset info display
            st.markdown("### Dataset Information")
            st.write(f"Dataset has {len(df)} rows and {len(df.columns)} columns.")
            st.write(f"Available columns: {', '.join(df.columns)}")

            # Show a small sample of the data
            st.markdown("### Data Sample")
            st.dataframe(df.head(5))

            st.info("For more advanced analysis, please configure the AI Assistant.")
        return

    # Display the full chat interface
    st.info("💬 Ask questions or request plots about the **loaded data**.")

    # Initialize chat messages if not already in session state
    if SESSION_CHAT_MESSAGES not in st.session_state:
        initial_message = f"Hi! I'm your banking compliance assistant. I can answer questions about your data ({len(df)} rows loaded). What would you like to know?"
        st.session_state[SESSION_CHAT_MESSAGES] = [{"role": "assistant", "content": initial_message}]

    # Display chat messages
    for message in st.session_state[SESSION_CHAT_MESSAGES]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Check if a chart object exists and display it
            if "chart" in message and message["chart"] is not None:
                try:
                    st.plotly_chart(message["chart"], use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not display chart: {e}")

    # Chat input
    prompt = st.chat_input(
        "Ask a question about the loaded data (e.g., 'Show me a bar chart of account types', 'How many dormant accounts?')..."
    )

    if prompt:
        # Add user message to chat history
        st.session_state[SESSION_CHAT_MESSAGES].append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response (text + optional chart)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Handle any errors during response generation
                    response_text, chart_obj = get_response_and_chart(prompt, df, llm)

                    # Display the response
                    st.markdown(response_text)

                    # Display chart if available
                    if chart_obj is not None:
                        try:
                            st.plotly_chart(chart_obj, use_container_width=True)
                        except Exception as chart_e:
                            st.warning(f"Could not display generated chart: {chart_e}")

                except Exception as e:
                    # Handle any unexpected errors
                    error_msg = f"I encountered an error while processing your request: {str(e)}"
                    st.error(error_msg)
                    response_text = error_msg
                    chart_obj = None

        # Add assistant response (and chart) to chat history
        assistant_response = {"role": "assistant", "content": response_text}
        if chart_obj is not None:
            assistant_response["chart"] = chart_obj  # Store the chart object
        st.session_state[SESSION_CHAT_MESSAGES].append(assistant_response)