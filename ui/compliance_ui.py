import streamlit as st
from datetime import datetime
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import SESSION_COLUMN_MAPPING
from agents.compliance import (
    detect_incomplete_contact, detect_flag_candidates,
    detect_ledger_candidates, detect_freeze_candidates,
    detect_transfer_candidates, log_flag_instructions,
    run_all_compliance_checks
)
from data.exporters import download_pdf_button, download_csv_button
from ai.llm import COMPLIANCE_SUMMARY_PROMPT, get_fallback_response


def render_compliance_analyzer(df, llm):
    """
    Render the Compliance Analyzer UI.

    Args:
        df (pandas.DataFrame): The account data to analyze
        llm: The LLM model for generating insights
    """
    st.subheader("🔒 Compliance Analysis Tasks")

    agent_options_compliance = [
        "📊 Summarized Compliance Detection",
        "📨 Contact Attempt Verification Agent",
        "🚩 Flag Dormant Candidate Agent",
        "📘 Dormant Ledger Review Agent",
        "❄️ Account Freeze Candidate Agent",
        "🏦 CBUAE Transfer Candidate Agent"
    ]

    selected_agent_compliance = st.selectbox(
        "Select Compliance Task or Summary",
        agent_options_compliance,
        key="compliance_agent_selector"
    )

    # Get threshold dates from session state
    general_threshold_date = st.session_state.get('general_threshold_date')
    freeze_threshold_date = st.session_state.get('freeze_threshold_date')
    cbuae_cutoff_date = st.session_state.get('cbuae_cutoff_date')

    # Check if thresholds are available
    if not all([general_threshold_date, freeze_threshold_date]):
        st.warning("Threshold dates not set. Please check sidebar settings.")
        return

    # Handle the summarized compliance option
    if selected_agent_compliance == "📊 Summarized Compliance Detection":
        render_summarized_compliance_analysis(df, general_threshold_date, freeze_threshold_date, cbuae_cutoff_date, llm)
    else:
        # Handle individual agent options
        render_individual_compliance_agent(df, selected_agent_compliance, general_threshold_date, freeze_threshold_date,
                                           cbuae_cutoff_date)


def render_summarized_compliance_analysis(df, general_threshold_date, freeze_threshold_date, cbuae_cutoff_date, llm):
    """Render the summarized compliance analysis UI."""
    st.subheader("📈 Summarized Compliance Detection Results")

    if st.button("📊 Run Summarized Compliance Analysis", key="run_summary_compliance_button"):
        with st.spinner("Running all compliance checks..."):
            results = run_all_compliance_checks(df, general_threshold_date, freeze_threshold_date, cbuae_cutoff_date)

        # Store results in session state for later reference
        st.session_state.compliance_summary_results = results

        # Display the numerical summary
        st.subheader("🔢 Numerical Summary")

        # Format days for display
        general_days = (datetime.now() - general_threshold_date).days
        freeze_days = (datetime.now() - freeze_threshold_date).days
        cbuae_date_str = cbuae_cutoff_date.strftime('%Y-%m-%d') if cbuae_cutoff_date else "Invalid date"

        st.metric(
            "Incomplete Contact Attempts",
            results['contact']['count'],
            help=results['contact']['desc']
        )
        st.metric(
            f"Flag Candidates (>={general_days} days inactive)",
            results['flag']['count'],
            help=results['flag']['desc']
        )
        st.metric(
            "Ledger Classification Needed",
            results['ledger']['count'],
            help=results['ledger']['desc']
        )
        st.metric(
            f"Freeze Candidates (>={freeze_days} days dormant)",
            results['freeze']['count'],
            help=results['freeze']['desc']
        )
        st.metric(
            f"CBUAE Transfer Candidates (Inactive before {cbuae_date_str})",
            results['transfer']['count'],
            help=results['transfer']['desc']
        )

        # Prepare input text for AI summary
        compliance_summary_input_text = (
            f"Compliance Analysis Findings ({results['total_accounts']} total accounts analyzed):\n"
            f"- {results['contact']['desc']}\n"
            f"- {results['flag']['desc']}\n"
            f"- {results['ledger']['desc']}\n"
            f"- {results['freeze']['desc']}\n"
            f"- {results['transfer']['desc']}"
        )

        # Add AI narrative summary if LLM is available
        st.subheader("📝 AI Compliance Summary")
        narrative_summary = compliance_summary_input_text  # Default to raw text

        if llm:
            try:
                with st.spinner("Generating AI Compliance Summary..."):
                    compliance_summary_prompt_template = PromptTemplate.from_template(COMPLIANCE_SUMMARY_PROMPT)
                    compliance_summary_chain = compliance_summary_prompt_template | llm | StrOutputParser()
                    compliance_narrative_summary = compliance_summary_chain.invoke({
                        "compliance_details": compliance_summary_input_text
                    })
                st.markdown(compliance_narrative_summary)
                st.session_state.compliance_narrative_summary = compliance_narrative_summary  # Store for PDF
            except Exception as llm_e:
                st.error(f"AI compliance summary generation failed: {llm_e}")
                fallback_summary = get_fallback_response("compliance_summary")
                st.warning(fallback_summary)
                st.text_area("Raw Compliance Findings:", compliance_summary_input_text, height=150)
                st.session_state.compliance_narrative_summary = f"{fallback_summary}\n\nRaw Findings:\n{compliance_summary_input_text}"
        else:
            fallback_summary = get_fallback_response("compliance_summary")
            st.warning(fallback_summary)
            st.text_area("Raw Compliance Findings:", compliance_summary_input_text, height=150)
            st.session_state.compliance_narrative_summary = f"{fallback_summary}\n\nRaw Findings:\n{compliance_summary_input_text}"

        # Export options
        st.subheader("⬇️ Export Summary")

        # Create report sections for PDF
        sections = [
            {
                "title": "Numerical Summary",
                "content": (
                    f"- {results['contact']['desc']}\n"
                    f"- {results['flag']['desc']}\n"
                    f"- {results['ledger']['desc']}\n"
                    f"- {results['freeze']['desc']}\n"
                    f"- {results['transfer']['desc']}"
                )
            },
            {
                "title": "Narrative Summary (AI Generated or Raw Findings)",
                "content": st.session_state.get('compliance_narrative_summary', "Summary not generated or AI failed.")
            }
        ]

        # Add download button
        download_pdf_button(
            "Compliance Analysis Summary Report",
            sections,
            f"compliance_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )


def render_individual_compliance_agent(df, selected_agent_compliance, general_threshold_date, freeze_threshold_date,
                                       cbuae_cutoff_date):
    """Render the UI for an individual compliance agent."""
    st.subheader(f"Agent Task Results: {selected_agent_compliance}")
    data_filtered = pd.DataFrame()
    agent_desc = "Select an agent above."
    agent_executed = False

    agent_mapping_compliance = {
        "📨 Contact Attempt Verification Agent": detect_incomplete_contact,
        "🚩 Flag Dormant Candidate Agent": detect_flag_candidates,
        "📘 Dormant Ledger Review Agent": detect_ledger_candidates,
        "❄️ Account Freeze Candidate Agent": detect_freeze_candidates,
        "🏦 CBUAE Transfer Candidate Agent": detect_transfer_candidates
    }

    if selected_agent_func := agent_mapping_compliance.get(selected_agent_compliance):
        with st.spinner(f"Running {selected_agent_compliance}..."):
            # Pass necessary args based on agent
            if selected_agent_compliance == "🚩 Flag Dormant Candidate Agent":
                data_filtered, count, agent_desc = selected_agent_func(df, general_threshold_date)

                # Store threshold days for logging
                general_threshold_days = (datetime.now() - general_threshold_date).days

            elif selected_agent_compliance == "❄️ Account Freeze Candidate Agent":
                data_filtered, count, agent_desc = selected_agent_func(df, freeze_threshold_date)

            elif selected_agent_compliance == "🏦 CBUAE Transfer Candidate Agent":
                # Handle potential invalid date from sidebar input
                if cbuae_cutoff_date is None:
                    data_filtered, count, agent_desc = pd.DataFrame(), 0, "Skipped due to invalid CBUAE cutoff date format."
                else:
                    data_filtered, count, agent_desc = selected_agent_func(df, cbuae_cutoff_date)

            else:  # "📨 Contact Attempt Verification Agent", "📘 Dormant Ledger Review Agent"
                data_filtered, count, agent_desc = selected_agent_func(df)

            agent_executed = True
            st.metric("Accounts Identified", count, help=agent_desc)

    if agent_executed:
        if not data_filtered.empty:
            st.success(f"{len(data_filtered)} accounts identified.")
            if st.checkbox(f"View first 15 detected accounts for '{selected_agent_compliance}'",
                           key=f"view_detected_{selected_agent_compliance.replace(' ', '_')}"):
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

            # Add buttons for next steps/logging based on agent type
            if selected_agent_compliance == "🚩 Flag Dormant Candidate Agent":
                if st.button("Log Flagging Instruction to DB (for Audit)", key="log_flag_instruction_compliance"):
                    if 'Account_ID' not in data_filtered.columns:
                        st.error("DataFrame does not have 'Account_ID' column. Cannot log.")
                    else:
                        try:
                            flagged_ids = data_filtered['Account_ID'].tolist()
                            success, message = log_flag_instructions(flagged_ids, selected_agent_compliance,
                                                                     general_threshold_days)

                            if success:
                                st.success(message)
                            else:
                                st.error(message)
                        except Exception as log_error:
                            st.error(f"Error logging flag instructions: {log_error}")

            # Add CSV export for all agent types
            st.subheader("⬇️ Export Data")
            download_csv_button(
                data_filtered,
                f"{selected_agent_compliance.replace(' ', '_').replace(':', '')}_accounts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            # Add specific guidance based on agent type
            if selected_agent_compliance == "📘 Dormant Ledger Review Agent":
                st.info("Review the accounts identified for manual classification in the dormant ledger.")

            elif selected_agent_compliance in ["❄️ Account Freeze Candidate Agent", "🏦 CBUAE Transfer Candidate Agent"]:
                action_type = "freeze" if "Freeze" in selected_agent_compliance else "transfer"
                st.info(
                    f"Accounts identified for potential {action_type} based on regulations. Review and take appropriate action according to your bank's policies.")

        elif len(data_filtered) == 0:
            st.info("No accounts matching the criteria were found.")