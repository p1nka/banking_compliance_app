import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import SESSION_COLUMN_MAPPING
from agents.compliance import (
    detect_incomplete_contact, detect_flag_candidates,
    detect_ledger_candidates, detect_freeze_candidates,
    detect_transfer_candidates, detect_foreign_currency_accounts,
    detect_safe_deposit_boxes, log_flag_instructions,
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
    st.subheader("🔒 UAE Dormant Accounts Compliance Analysis")

    agent_options_compliance = [
        "📊 Summarized Compliance Detection",
        "📨 Contact Attempt Verification Agent",
        "🚩 Flag Dormant Candidate Agent",
        "📘 Article 3 Notification Process Agent",
        "❄️ Account Freeze Candidate Agent",
        "🏦 CBUAE Transfer Candidate Agent",
        "💱 Foreign Currency Conversion Agent",
        "🔐 Safe Deposit Box Agent"
    ]

    selected_agent_compliance = st.selectbox(
        "Select Compliance Task or Summary",
        agent_options_compliance,
        key="compliance_agent_selector"
    )

    # Use regulation-defined thresholds
    three_years_ago = datetime.now() - timedelta(days=365 * 3)

    st.sidebar.subheader("UAE Regulation Information")
    st.sidebar.info(
        "As per UAE Central Bank Regulation No. 1/2020:\n"
        "- Standard Accounts: 3 years inactivity for dormancy\n"
        "- Payment Instruments: 1 year for dormancy\n"
        "- Article 3 Process: 3 months notification period\n"
        "- Transfer to Central Bank: After 5 years of dormancy"
    )

    # Store threshold dates in session state for reference
    st.session_state['general_threshold_date'] = three_years_ago

    # Handle the summarized compliance option
    if selected_agent_compliance == "📊 Summarized Compliance Detection":
        render_summarized_compliance_analysis(df, llm)
    else:
        # Handle individual agent options
        render_individual_compliance_agent(df, selected_agent_compliance)


def render_summarized_compliance_analysis(df, llm):
    """Render the summarized compliance analysis UI."""
    st.subheader("📈 Summarized Compliance Detection Results")

    if st.button("📊 Run Summarized Compliance Analysis", key="run_summary_compliance_button"):
        with st.spinner("Running all compliance checks..."):
            results = run_all_compliance_checks(df)

        # Store results in session state for later reference
        st.session_state.compliance_summary_results = results

        # Display the numerical summary
        st.subheader("🔢 Numerical Summary")

        st.metric(
            "Incomplete Contact Attempts",
            results['contact']['count'],
            help=results['contact']['desc']
        )
        st.metric(
            "Flag Candidates (3+ years inactive)",
            results['flag']['count'],
            help=results['flag']['desc']
        )
        st.metric(
            "Article 3 Process Required",
            results['ledger']['count'],
            help=results['ledger']['desc']
        )
        st.metric(
            "Account Freeze Candidates",
            results['freeze']['count'],
            help=results['freeze']['desc']
        )
        st.metric(
            "CBUAE Transfer Candidates",
            results['transfer']['count'],
            help=results['transfer']['desc']
        )
        st.metric(
            "Foreign Currency Accounts Requiring Conversion",
            results.get('foreign_currency', {}).get('count', 0),
            help=results.get('foreign_currency', {}).get('desc', "Foreign currency accounts requiring AED conversion")
        )
        st.metric(
            "Dormant Safe Deposit Boxes",
            results.get('safe_deposit', {}).get('count', 0),
            help=results.get('safe_deposit', {}).get('desc', "Safe deposit boxes requiring action")
        )

        # Prepare input text for AI summary
        compliance_summary_input_text = (
            f"Compliance Analysis Findings based on UAE Central Bank Regulation No. 1/2020 ({results['total_accounts']} accounts analyzed):\n"
            f"- {results['contact']['desc']}\n"
            f"- {results['flag']['desc']}\n"
            f"- {results['ledger']['desc']}\n"
            f"- {results['freeze']['desc']}\n"
            f"- {results['transfer']['desc']}\n"
        )

        if 'foreign_currency' in results:
            compliance_summary_input_text += f"- {results['foreign_currency']['desc']}\n"

        if 'safe_deposit' in results:
            compliance_summary_input_text += f"- {results['safe_deposit']['desc']}\n"

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
                "content": compliance_summary_input_text
            },
            {
                "title": "Narrative Summary (AI Generated or Raw Findings)",
                "content": st.session_state.get('compliance_narrative_summary', "Summary not generated or AI failed.")
            },
            {
                "title": "Regulatory Framework",
                "content": "This report is based on the UAE Central Bank Dormant Accounts Regulation (Circular No. 1/2020) dated January 15, 2020, which establishes requirements for handling dormant accounts and unclaimed balances."
            }
        ]

        # Add download button
        download_pdf_button(
            "Compliance Analysis Summary Report",
            sections,
            f"compliance_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )


def render_individual_compliance_agent(df, selected_agent_compliance):
    """Render the UI for an individual compliance agent."""
    st.subheader(f"Agent Task Results: {selected_agent_compliance}")
    data_filtered = pd.DataFrame()
    agent_desc = "Select an agent above."
    agent_executed = False

    # Get threshold dates
    three_years_ago = datetime.now() - timedelta(days=365 * 3)

    agent_mapping_compliance = {
        "📨 Contact Attempt Verification Agent": detect_incomplete_contact,
        "🚩 Flag Dormant Candidate Agent": detect_flag_candidates,
        "📘 Article 3 Notification Process Agent": detect_ledger_candidates,
        "❄️ Account Freeze Candidate Agent": detect_freeze_candidates,
        "🏦 CBUAE Transfer Candidate Agent": detect_transfer_candidates,
        "💱 Foreign Currency Conversion Agent": detect_foreign_currency_accounts,
        "🔐 Safe Deposit Box Agent": detect_safe_deposit_boxes
    }

    if selected_agent_func := agent_mapping_compliance.get(selected_agent_compliance):
        with st.spinner(f"Running {selected_agent_compliance}..."):
            # Pass necessary args based on agent
            if selected_agent_compliance in ["🚩 Flag Dormant Candidate Agent", "❄️ Account Freeze Candidate Agent"]:
                data_filtered, count, agent_desc = selected_agent_func(df, three_years_ago)
                # Store threshold days for logging
                general_threshold_days = (datetime.now() - three_years_ago).days
            else:  # Other agents don't require threshold dates
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
            if selected_agent_compliance == "📘 Article 3 Notification Process Agent":
                st.info(
                    "These accounts require the Article 3 notification process. Per UAE Central Bank Regulation, you should attempt to contact customers through multiple channels and wait 3 months before transferring to dormant ledger.")

            elif selected_agent_compliance == "❄️ Account Freeze Candidate Agent":
                st.info(
                    "Accounts identified for potential freeze based on UAE Central Bank regulations. These accounts should be monitored for unauthorized operations and physical/electronic statements suppressed.")

            elif selected_agent_compliance == "🏦 CBUAE Transfer Candidate Agent":
                st.info(
                    "Accounts identified for transfer to the Central Bank of UAE after 5 years of dormancy. No fees other than agreed should be levied on these accounts.")

            elif selected_agent_compliance == "💱 Foreign Currency Conversion Agent":
                st.info(
                    "Foreign currency accounts must be converted to AED at the Bank's published customer rates before transfer to the Central Bank as per Regulation Article 8.5.")

            elif selected_agent_compliance == "🔐 Safe Deposit Box Agent":
                st.info(
                    "Safe deposit boxes with unpaid fees for over 3 years. Per Regulation Article 2.6, apply to the Court to appoint a person to supervise opening of the box and provide direction regarding disposal of contents.")

        elif len(data_filtered) == 0:
            st.info("No accounts matching the criteria were found.")