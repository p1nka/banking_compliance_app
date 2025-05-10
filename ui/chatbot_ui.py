import streamlit as st
from ai.chatbot import get_response_and_chart
from config import SESSION_CHAT_MESSAGES


def render_chatbot(df, llm):
    """
    Render the chatbot UI.

    Args:
        df (pandas.DataFrame): The account data to analyze
        llm: The LLM model for generating responses
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

    # Display chat interface with full functionality
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