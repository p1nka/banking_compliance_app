import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import SESSION_COLUMN_MAPPING
from agents.dormant import (
    check_safe_deposit, check_investment_inactivity,
    check_fixed_deposit_inactivity, check_general_inactivity,
    check_unreachable_dormant, run_all_dormant_checks
)
from database.operations import save_summary_to_db
from data.exporters import download_pdf_button, download_csv_button
from ai.llm import (
    get_fallback_response,
    DORMANT_SUMMARY_PROMPT,
    OBSERVATION_PROMPT,
    TREND_PROMPT,
    NARRATION_PROMPT,
    ACTION_PROMPT
)


def render_dormant_analyzer(df, llm):
    """
    Render the Dormant Account Analyzer UI.

    Args:
        df (pandas.DataFrame): The account data to analyze
        llm: The LLM model for generating insights
    """
    st.subheader("🏦 Dormant Account Analysis")
    threshold = datetime.now() - timedelta(days=3 * 365)  # 3 years inactivity threshold

    agent_option = st.selectbox(
        "🧭 Choose Dormant Detection Agent",
        [
            "📊 Summarized Dormant Analysis",
            "🔐 Safe Deposit Box Agent",
            "💼 Investment Inactivity Agent",
            "🏦 Fixed Deposit Agent",
            "📉 3-Year General Inactivity Agent",
            "📵 Unreachable + No Active Accounts Agent"
        ],
        key="dormant_agent_selector"
    )

    # Handle the summarized dormant analysis option
    if agent_option == "📊 Summarized Dormant Analysis":
        render_summarized_dormant_analysis(df, threshold, llm)
    else:
        # Handle individual agent options
        render_individual_dormant_agent(df, agent_option, threshold, llm)


def render_summarized_dormant_analysis(df, threshold, llm):
    """Render the summarized dormant analysis UI."""
    st.subheader("📈 Summarized Dormant Analysis Results")

    if st.button("📊 Run Summarized Dormant Analysis", key="run_summary_dormant_button"):
        with st.spinner("Running all dormant checks..."):
            results = run_all_dormant_checks(df, threshold)

        # Store results in session state for later reference
        st.session_state.dormant_summary_results = results

        # Display the numerical summary
        st.subheader("🔢 Numerical Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Uncontacted Safe Deposit (>3y)",
                results["sd"]["count"],
                help=results["sd"]["desc"]
            )
            st.metric(
                "General Inactivity (>3y)",
                results["gen"]["count"],
                help=results["gen"]["desc"]
            )
        with col2:
            st.metric(
                "Uncontacted Investment (>3y)",
                results["inv"]["count"],
                help=results["inv"]["desc"]
            )
            st.metric(
                "Unreachable & 'Dormant'",
                results["unr"]["count"],
                help=results["unr"]["desc"]
            )
        with col3:
            st.metric(
                "Fixed Deposit Inactivity (>3y)",
                results["fd"]["count"],
                help=results["fd"]["desc"]
            )

        # Prepare input text for AI summary
        summary_input_text = (
            f"Dormant Analysis Findings ({results['total_accounts']} total accounts analyzed, threshold >3 years inactive):\n"
            f"- {results['sd']['desc']}\n"
            f"- {results['inv']['desc']}\n"
            f"- {results['fd']['desc']}\n"
            f"- {results['gen']['desc']}\n"
            f"- {results['unr']['desc']}"
        )

        st.subheader("📝 Narrative Summary")
        narrative_summary = summary_input_text  # Default to raw data in case AI fails

        if llm:
            try:
                with st.spinner("Generating AI Summary..."):
                    # Use the predefined summary prompt template
                    summary_prompt_template = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
                    summary_chain = summary_prompt_template | llm | StrOutputParser()
                    narrative_summary = summary_chain.invoke({
                        "analysis_details": summary_input_text
                    })
                    st.markdown(narrative_summary)
                    st.session_state.dormant_narrative_summary = narrative_summary  # Store for PDF
            except Exception as llm_e:
                st.error(f"AI summary generation failed: {llm_e}")
                fallback_summary = get_fallback_response("dormant_summary")
                st.warning(fallback_summary)
                st.text_area("Raw Findings:", summary_input_text, height=150)
                st.session_state.dormant_narrative_summary = f"{fallback_summary}\n\nRaw Findings:\n{summary_input_text}"
        else:
            fallback_summary = get_fallback_response("dormant_summary")
            st.warning(fallback_summary)
            st.text_area("Raw Findings:", summary_input_text, height=150)
            st.session_state.dormant_narrative_summary = f"{fallback_summary}\n\nRaw Findings:\n{summary_input_text}"

        # Export options
        st.subheader("⬇️ Export Summary")

        # Create report sections for PDF
        sections = [
            {
                "title": "Numerical Summary",
                "content": f"- {results['sd']['desc']}\n- {results['inv']['desc']}\n- {results['fd']['desc']}\n- {results['gen']['desc']}\n- {results['unr']['desc']}"
            },
            {
                "title": "Narrative Summary (AI Generated or Raw Findings)",
                "content": st.session_state.get('dormant_narrative_summary', "Summary not generated or AI failed.")
            }
        ]

        # Add download button
        download_pdf_button(
            "Dormant Account Analysis Summary Report",
            sections,
            f"dormant_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )


def render_individual_dormant_agent(df, agent_option, threshold, llm):
    """Render the UI for an individual dormant agent."""
    st.subheader(f"Agent Task Results: {agent_option}")
    data_filtered = pd.DataFrame()
    agent_desc = "Select an agent above."
    agent_executed = False

    agent_mapping = {
        "🔐 Safe Deposit Box Agent": check_safe_deposit,
        "💼 Investment Inactivity Agent": check_investment_inactivity,
        "🏦 Fixed Deposit Agent": check_fixed_deposit_inactivity,
        "📉 3-Year General Inactivity Agent": check_general_inactivity,
        "📵 Unreachable + No Active Accounts Agent": check_unreachable_dormant
    }

    if selected_agent := agent_mapping.get(agent_option):
        with st.spinner(f"Running {agent_option}..."):
            # Pass necessary args based on agent
            if agent_option in [
                "🔐 Safe Deposit Box Agent",
                "💼 Investment Inactivity Agent",
                "🏦 Fixed Deposit Agent",
                "📉 3-Year General Inactivity Agent"
            ]:
                data_filtered, count, agent_desc = selected_agent(df, threshold)
            else:  # "📵 Unreachable + No Active Accounts Agent"
                data_filtered, count, agent_desc = selected_agent(df)
            agent_executed = True

        st.metric("Accounts Identified", count, help=agent_desc)
        st.markdown(f"**Agent Description:** {agent_desc}")

    if agent_executed:
        if not data_filtered.empty:
            st.success(f"{len(data_filtered)} accounts identified.")
            if st.checkbox(f"View first 15 detected accounts for '{agent_option}'",
                           key=f"view_detected_{agent_option.replace(' ', '_')}"):
                # Display the DataFrame with original column names if available
                display_df = data_filtered.head(15).copy()
                if SESSION_COLUMN_MAPPING in st.session_state and st.session_state[SESSION_COLUMN_MAPPING]:
                    try:
                        # Create a display mapping that only includes columns present in the data
                        display_columns_mapping = {
                            std_col: st.session_state[SESSION_COLUMN_MAPPING].get(std_col, std_col)
                            for std_col in display_df.columns
                        }
                        display_df.rename(columns=display_columns_mapping, inplace=True)
                    except Exception as e:
                        st.warning(f"Could not display original column names: {e}")

                st.dataframe(display_df)

            # Generate insights with LLM if available
            if llm:
                sample_size = min(15, len(data_filtered))
                sample_data_csv = data_filtered.sample(n=sample_size).to_csv(index=False)

                # Initialize the prompt templates with our predefined prompts
                observation_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                trend_prompt = PromptTemplate.from_template(TREND_PROMPT)

                if st.button(f"Generate Insights for '{agent_option}'",
                             key=f"generate_insights_{agent_option.replace(' ', '_')}"):
                    try:
                        with st.spinner("Running insight agents..."):
                            # Setup the LLM chains
                            output_parser = StrOutputParser()
                            observation_chain = observation_prompt | llm | output_parser
                            trend_chain = trend_prompt | llm | output_parser

                            # Generate observations and trends
                            obs_output = observation_chain.invoke({"data": sample_data_csv})
                            trend_output = trend_chain.invoke({"data": sample_data_csv})

                            # Generate summary and actions
                            narration_prompt = PromptTemplate.from_template(NARRATION_PROMPT)
                            action_prompt = PromptTemplate.from_template(ACTION_PROMPT)

                            narration_chain = narration_prompt | llm | output_parser
                            action_chain = action_prompt | llm | output_parser

                            final_insight = narration_chain.invoke({"observation": obs_output, "trend": trend_output})
                            action_output = action_chain.invoke({"observation": obs_output, "trend": trend_output})

                        # Store insights in session state
                        st.session_state[f'{agent_option}_insights'] = {
                            'observation': obs_output,
                            'trend': trend_output,
                            'summary': final_insight,
                            'actions': action_output
                        }

                        # Save to DB log
                        try:
                            if save_summary_to_db(obs_output, trend_output, final_insight, action_output):
                                st.success("Insights saved to insight log.")
                            else:
                                st.error("Failed to save insights to DB log.")
                        except Exception as e:
                            st.error(f"Error saving insights to DB: {e}")

                    except Exception as insight_error:
                        st.error(f"Error generating insights: {insight_error}")
                        # Use fallback responses
                        obs_output = get_fallback_response("observation")
                        trend_output = get_fallback_response("trend")
                        final_insight = get_fallback_response("narration")
                        action_output = get_fallback_response("action")

                        # Store fallback insights
                        st.session_state[f'{agent_option}_insights'] = {
                            'observation': obs_output,
                            'trend': trend_output,
                            'summary': final_insight,
                            'actions': action_output
                        }

                        st.warning("Using fallback insights due to AI error.")

                # Display insights if they exist in session state
                if f'{agent_option}_insights' in st.session_state:
                    insights = st.session_state[f'{agent_option}_insights']
                    with st.expander("🔍 Observation Insight"):
                        st.markdown(insights['observation'])
                    with st.expander("📊 Trend Insight"):
                        st.markdown(insights['trend'])
                    with st.expander("📌 CXO Summary"):
                        st.markdown(insights['summary'])
                    with st.expander("🚀 Recommended Actions"):
                        st.markdown(insights['actions'])

                    # PDF export for individual agent insights
                    st.subheader("⬇️ Export Insights")
                    sections = [
                        {"title": "Observations", "content": insights['observation']},
                        {"title": "Trends", "content": insights['trend']},
                        {"title": "Executive Summary", "content": insights['summary']},
                        {"title": "Recommended Actions", "content": insights['actions']}
                    ]

                    download_pdf_button(
                        f"{agent_option} - Analysis Report",
                        sections,
                        f"{agent_option.replace(' ', '_').replace(':', '')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    )

                    # Add CSV download for identified accounts
                    download_csv_button(
                        data_filtered,
                        f"{agent_option.replace(' ', '_').replace(':', '')}_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )

            elif len(data_filtered) > 0:
                st.info(f"AI Assistant not available to generate insights.")

                # Still provide CSV download even without AI
                st.subheader("⬇️ Export Data")
                download_csv_button(
                    data_filtered,
                    f"{agent_option.replace(' ', '_').replace(':', '')}_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )

        elif len(data_filtered) == 0:
            st.info("No accounts matching the criteria were found.")