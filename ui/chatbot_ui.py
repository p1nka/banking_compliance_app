import streamlit as st
import pandas as pd
from ai.chatbot import get_response_and_chart, display_chat_interface
from ai.visualizations import generate_plot, create_insights_chart
from config import SESSION_CHAT_MESSAGES


def render_chatbot(df, llm):
    """
    Render the enhanced chatbot UI with more visualization capabilities.

    Args:
        df (pandas.DataFrame): The account data to analyze
        llm: The LLM model for generating responses
    """
    st.header("Banking Compliance Chatbot")

    if df is None or df.empty:
        st.error("No data available. Please upload and process data first.")
        return

    # Display data summary sidebar
    with st.sidebar:
        st.subheader("📊 Dataset Overview")
        st.metric("Total Records", len(df))
        st.metric("Columns", len(df.columns))

        # Show quick insights if data is available
        if not df.empty:
            st.subheader("Quick Insights")

            # Create insight options
            if st.checkbox("Show Column Statistics"):
                selected_col = st.selectbox("Select column:", df.columns)

                if selected_col:
                    col_stats = pd.DataFrame({
                        'Statistic': ['Count', 'Unique Values', 'Missing Values', 'Top Value'],
                        'Value': [
                            len(df[selected_col]),
                            df[selected_col].nunique(),
                            df[selected_col].isna().sum(),
                            str(df[selected_col].value_counts().index[0]) if not df[selected_col].empty else "N/A"
                        ]
                    }).set_index('Statistic')

                    st.dataframe(col_stats)

                    # Add quick visualization if appropriate
                    if df[selected_col].nunique() < 15 and not pd.api.types.is_numeric_dtype(df[selected_col]):
                        try:
                            chart = create_insights_chart(
                                df,
                                labels=selected_col,
                                chart_type='pie',
                                title=f"Distribution of {selected_col}"
                            )
                            if chart:
                                st.plotly_chart(chart, use_container_width=True)
                        except Exception as e:
                            st.info(f"Could not create chart: {e}")

            # Add visualization suggestions
            st.subheader("Visualization Ideas")
            viz_suggestions = [
                "Show me a bar chart of account types",
                "Create a pie chart of dormant vs active accounts",
                "Plot the distribution of account balances",
                "Show the timeline of customer last activity dates",
                "Compare account status across different branches"
            ]

            if st.button("🔎 Suggest Visualizations"):
                suggestion = viz_suggestions[hash(df.shape[0]) % len(viz_suggestions)]
                st.info(f"Try asking: \"{suggestion}\"")

    # Check if LLM is available
    if llm is None:
        st.warning("⚠️ AI Assistant not available. Chat functionality will be limited.")
        st.info(
            "To enable full chatbot functionality, please configure your GROQ API key in secrets.toml or as an environment variable.")

        # Offer a simplified view with basic data exploration
        st.subheader("Basic Data Explorer")

        # Simple tab interface for basic exploration
        basic_tabs = st.tabs(["Data Sample", "Column Info", "Simple Visualizations"])

        with basic_tabs[0]:
            st.dataframe(df.head(10))

        with basic_tabs[1]:
            st.write("### Available Columns")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)

        with basic_tabs[2]:
            st.write("### Simple Data Visualization")
            viz_col = st.selectbox("Select a column to visualize:", df.columns)
            viz_type = st.selectbox("Select visualization type:", ["Bar Chart", "Pie Chart", "Histogram"])

            if viz_col and viz_type:
                try:
                    if viz_type == "Bar Chart":
                        if pd.api.types.is_numeric_dtype(df[viz_col]):
                            st.warning(f"{viz_col} is numeric. Using histogram instead.")
                            chart = create_insights_chart(df, labels=viz_col, chart_type='histogram',
                                                          title=f"Distribution of {viz_col}")
                        else:
                            chart = create_insights_chart(df, labels=viz_col, chart_type='bar',
                                                          title=f"Count of {viz_col}")

                    elif viz_type == "Pie Chart":
                        if df[viz_col].nunique() > 15:
                            st.warning(f"Too many unique values in {viz_col} for a pie chart. Showing top 10.")
                            # Get top 10 values and group the rest as "Other"
                            top_values = df[viz_col].value_counts().nlargest(10).index
                            df_temp = df.copy()
                            df_temp[viz_col] = df_temp[viz_col].apply(lambda x: x if x in top_values else 'Other')
                            chart = create_insights_chart(df_temp, labels=viz_col, chart_type='pie',
                                                          title=f"Distribution of {viz_col} (Top 10)")
                        else:
                            chart = create_insights_chart(df, labels=viz_col, chart_type='pie',
                                                          title=f"Distribution of {viz_col}")

                    elif viz_type == "Histogram":
                        if not pd.api.types.is_numeric_dtype(df[viz_col]):
                            st.warning(f"{viz_col} is not numeric. Using bar chart instead.")
                            chart = create_insights_chart(df, labels=viz_col, chart_type='bar',
                                                          title=f"Count of {viz_col}")
                        else:
                            chart = create_insights_chart(df, labels=viz_col, chart_type='histogram',
                                                          title=f"Distribution of {viz_col}")

                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        st.error("Could not create visualization.")

                except Exception as viz_error:
                    st.error(f"Visualization error: {viz_error}")

        st.info("For more advanced analysis and interactive chat, please configure the AI Assistant.")
        return

    # Use the imported display_chat_interface function from chatbot.py for the full chat experience
    display_chat_interface(df, llm)

    # Add advanced features section
    st.subheader("Advanced Features")
    advanced_expander = st.expander("🔧 Advanced Options", expanded=False)

    with advanced_expander:
        st.write("### Chat History Options")

        # Option to clear chat history
        if st.button("🗑️ Clear Chat History"):
            if SESSION_CHAT_MESSAGES in st.session_state:
                initial_message = f"Hi! I'm your banking compliance assistant. Chat history has been cleared. What would you like to know about your data ({len(df)} rows loaded)?"
                st.session_state[SESSION_CHAT_MESSAGES] = [{"role": "assistant", "content": initial_message}]
                st.success("Chat history cleared!")
                st.experimental_rerun()

        # Option to export chat
        if SESSION_CHAT_MESSAGES in st.session_state and len(st.session_state[SESSION_CHAT_MESSAGES]) > 1:
            chat_text = "\n\n".join([
                f"**{msg['role'].title()}**: {msg['content']}"
                for msg in st.session_state[SESSION_CHAT_MESSAGES]
            ])

            st.download_button(
                label="📥 Export Chat History",
                data=chat_text,
                file_name=f"banking_compliance_chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )

        # Add sample question suggestions
        st.write("### Sample Questions")
        question_categories = {
            "Data Exploration": [
                "What's the distribution of account types in the data?",
                "How many dormant accounts do we have?",
                "What's the average balance of dormant accounts?"
            ],
            "Visualizations": [
                "Show me a pie chart of account statuses",
                "Create a histogram of account balances",
                "Generate a bar chart of account types by dormancy status"
            ],
            "Compliance Analysis": [
                "Which accounts haven't had activity in more than 2 years?",
                "What percentage of accounts are flagged for compliance review?",
                "Show accounts with incomplete contact attempts"
            ]
        }

        question_category = st.selectbox("Select a category:", list(question_categories.keys()))

        if question_category:
            questions = question_categories[question_category]
            selected_question = st.selectbox("Select a sample question:", questions)

            if st.button("Ask Selected Question"):
                # Add to session state so the chat interface will pick it up
                if SESSION_CHAT_MESSAGES in st.session_state:
                    st.session_state[SESSION_CHAT_MESSAGES].append({"role": "user", "content": selected_question})
                    st.experimental_rerun()

    # Add helpful tips section at the bottom
    with st.expander("📝 Tips for using the chatbot", expanded=False):
        st.markdown("""
        ### How to get the best results from the AI Chatbot:

        - Be specific in your questions (e.g., "How many accounts with balance over $1000 are dormant?" instead of "Tell me about dormant accounts")
        - For visualizations, specify the chart type and data you want to see (e.g., "Create a bar chart showing account types")
        - You can ask for analysis like "Compare active vs dormant accounts by balance"
        - Use the chat history to build on previous questions
        - If you get an error, try rephrasing your question
        - The AI works best with the data columns available - check the dataset overview if unsure
        """)