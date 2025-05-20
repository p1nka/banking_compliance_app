import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
# Assuming your backend compliance.py is in an 'agents' directory
from agents.compliance import (
    run_all_compliance_checks,
    detect_incomplete_contact,
    detect_flag_candidates,
    log_flag_instructions,  # Important for the flag candidates agent
    detect_ledger_candidates,
    detect_freeze_candidates,
    detect_transfer_candidates_to_cb,  # Renamed from detect_transfer_candidates
    detect_foreign_currency_conversion_needed,  # Renamed
    detect_sdb_court_application_needed,  # Renamed from detect_safe_deposit_boxes
    detect_unclaimed_payment_instruments_ledger,  # Renamed
    detect_claim_processing_pending,  # Renamed from detect_claim_candidates
    generate_annual_cbuae_report_summary,  # Renamed
    check_record_retention_compliance  # Renamed
)
# Assuming these utility modules exist
from data.exporters import download_pdf_button, download_csv_button
from ai.llm import (
    get_fallback_response,
    COMPLIANCE_SUMMARY_PROMPT,
    OBSERVATION_PROMPT,
    TREND_PROMPT,
    NARRATION_PROMPT,
    ACTION_PROMPT
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# from database.operations import save_summary_to_db # If you intend to save insights

# --- Main Rendering Function for Compliance UI ---
def render_compliance_analyzer(df, agent_name_input, llm):
    """
    Main function to render the Compliance Audit Analyzer UI.
    """
    st.header("🛡️ Compliance Audit Analyzer (CBUAE)")

    agent_options_compliance = [
        "📊 Summarized Compliance Audit (All Checks)",
        "--- Individual Agent Checks ---",
        "CONTACT: Incomplete Contact Attempts",
        "FLAG: Flag Candidates (Not Yet Flagged)",
        "LEDGER: Internal Ledger Candidates (Art. 3.5)",
        "FREEZE: Statement Freeze Needed (Art. 7.3)",
        "CB_TRANSFER: CBUAE Transfer Candidates (Art. 8)",
        "FX_CONV: Foreign Currency Conversion for CB Transfer",
        "SDB_APP: SDB Court Application Needed",
        "PI_LEDGER: Unclaimed Instruments for Internal Ledger",
        "CLAIMS: Claims Processing Pending (>1 Month)",
        "ANNUAL_RPT: Annual CBUAE Report Summary",
        "RETENTION: Record Retention Compliance"
    ]

    selected_agent_compliance = st.selectbox(
        "Select Compliance Audit Task",
        agent_options_compliance,
        key="compliance_agent_selector_ui"
    )

    if selected_agent_compliance == "📊 Summarized Compliance Audit (All Checks)":
        render_summarized_compliance_audit_view(df, agent_name_input, llm)
    elif selected_agent_compliance != "--- Individual Agent Checks ---":
        render_individual_compliance_agent_view(df, selected_agent_compliance, agent_name_input, llm)


# --- Summarized View ---
def render_summarized_compliance_audit_view(df, agent_name_input, llm):
    st.subheader("📈 Summarized Compliance Audit Results")

    if st.button("🚀 Run Summarized Compliance Audit", key="run_summary_compliance_audit_button"):
        with st.spinner("Running all compliance audit checks..."):
            results = run_all_compliance_checks(df.copy(), agent_name=agent_name_input)
        st.session_state.compliance_summary_results_ui = results
        st.toast("Summarized compliance audit complete!", icon="✅")

    if 'compliance_summary_results_ui' in st.session_state:
        results = st.session_state.compliance_summary_results_ui

        st.markdown(f"**Total Accounts Processed:** `{results.get('total_accounts_processed', 'N/A')}`")

        flag_log_status = results.get("flag_logging_status", {})
        if flag_log_status.get("status"):
            st.success(f"Flagging Log: {flag_log_status.get('message', 'Logged.')}")
        else:
            st.warning(f"Flagging Log: {flag_log_status.get('message', 'Not logged or issue.')}")

        st.subheader("Key Compliance Metrics")
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Incomplete Contact", results.get('incomplete_contact', {}).get('count', 0),
                    help=results.get('incomplete_contact', {}).get('desc', ""))
        col1.metric("Flag Candidates", results.get('flag_candidates', {}).get('count', 0),
                    help=results.get('flag_candidates', {}).get('desc', ""))
        col2.metric("Internal Ledger Candidates", results.get('ledger_candidates_internal', {}).get('count', 0),
                    help=results.get('ledger_candidates_internal', {}).get('desc', ""))
        col2.metric("Statement Freeze Needed", results.get('statement_freeze_needed', {}).get('count', 0),
                    help=results.get('statement_freeze_needed', {}).get('desc', ""))
        col3.metric("CBUAE Transfer Candidates", results.get('transfer_candidates_cb', {}).get('count', 0),
                    help=results.get('transfer_candidates_cb', {}).get('desc', ""))
        col3.metric("Claims Pending (>1 Month)", results.get('claims_processing_pending', {}).get('count', 0),
                    help=results.get('claims_processing_pending', {}).get('desc', ""))

        # --- Chart for Summary ---
        st.subheader("Visual Insights")
        compliance_counts_data = {
            "Incomplete Contact": results.get('incomplete_contact', {}).get('count', 0),
            "Flag Candidates": results.get('flag_candidates', {}).get('count', 0),
            "Ledger Internal": results.get('ledger_candidates_internal', {}).get('count', 0),
            "Freeze Needed": results.get('statement_freeze_needed', {}).get('count', 0),
            "CB Transfer": results.get('transfer_candidates_cb', {}).get('count', 0),
            "Claims Pending": results.get('claims_processing_pending', {}).get('count', 0),
        }
        compliance_counts_df = pd.DataFrame(list(compliance_counts_data.items()),
                                            columns=['Category', 'Count']).set_index('Category')
        if not compliance_counts_df.empty:
            st.bar_chart(compliance_counts_df, height=300)
        else:
            st.info("No data for compliance category chart.")

        summary_input_text = f"Compliance Audit Report (Total Processed: {results.get('total_accounts_processed')})\n"
        for key, val_dict in results.items():
            if isinstance(val_dict, dict) and "desc" in val_dict and "count" in val_dict:
                if not val_dict["desc"].startswith("(Skipped"):
                    summary_input_text += f"- {val_dict['desc']}\n"
            elif key == "flag_logging_status":
                summary_input_text += f"- Flagging Log: {val_dict.get('message')}\n"
            elif key == "record_retention_check":
                summary_input_text += f"- {val_dict.get('desc')}\n"

        st.subheader("📝 AI Generated Summary & Insights")
        if llm:
            try:
                with st.spinner("Generating AI summary for all compliance checks..."):
                    prompt_template = PromptTemplate.from_template(COMPLIANCE_SUMMARY_PROMPT)
                    chain = prompt_template | llm | StrOutputParser()
                    ai_summary = chain.invoke({"compliance_details": summary_input_text})
                st.markdown(ai_summary)
                st.session_state.compliance_ai_summary_text_ui = ai_summary
            except Exception as e:
                st.error(f"AI summary generation failed: {e}")
                st.session_state.compliance_ai_summary_text_ui = get_fallback_response(
                    "compliance_summary") + f"\n\nRaw Data:\n{summary_input_text}"
                st.warning(st.session_state.compliance_ai_summary_text_ui)
        else:
            st.warning("LLM not available. Displaying raw findings.")
            st.text_area("Raw Findings for Summary", summary_input_text, height=200)
            st.session_state.compliance_ai_summary_text_ui = summary_input_text

        # --- Export Options for Summary ---
        st.subheader("⬇️ Export Summarized Report")
        summary_report_sections = [
            {"title": "Compliance Audit Overview", "content": summary_input_text},
            {"title": "AI Generated Summary",
             "content": st.session_state.get("compliance_ai_summary_text_ui", "AI Summary not generated.")}
        ]
        download_pdf_button("Compliance_Audit_Summary_Report", summary_report_sections, "compliance_summary_report.pdf")

        with st.expander("View Raw Results for All Compliance Checks"):
            st.json(results, expanded=False)


# --- Individual Agent View ---
def render_individual_compliance_agent_view(df, selected_agent_key, agent_name_input, llm):
    st.subheader(f"🔍 Results for: {selected_agent_key}")
    three_years_ago = datetime.now() - timedelta(days=3 * 365)  # For agents needing this

    # Agent mapping (key from dropdown to function and description)
    agent_functions_compliance = {
        "CONTACT: Incomplete Contact Attempts": (detect_incomplete_contact,
                                                 "Detects accounts with incomplete contact attempts (Art. 3.1)."),
        "FLAG: Flag Candidates (Not Yet Flagged)": (detect_flag_candidates,
                                                    "Detects accounts inactive over threshold, not yet flagged dormant (Art. 2)."),
        "LEDGER: Internal Ledger Candidates (Art. 3.5)": (detect_ledger_candidates,
                                                          "Detects accounts for internal 'dormant accounts ledger' (Art. 3.5)."),
        "FREEZE: Statement Freeze Needed (Art. 7.3)": (detect_freeze_candidates,
                                                       "Detects dormant accounts requiring statement suppression (Art. 7.3)."),
        "CB_TRANSFER: CBUAE Transfer Candidates (Art. 8)": (detect_transfer_candidates_to_cb,
                                                            "Detects dormant accounts/balances for CBUAE transfer (Art. 8)."),
        "FX_CONV: Foreign Currency Conversion for CB Transfer": (detect_foreign_currency_conversion_needed,
                                                                 "Detects foreign currency items for CBUAE transfer requiring AED conversion (Art. 8.5)."),
        "SDB_APP: SDB Court Application Needed": (detect_sdb_court_application_needed,
                                                  "Detects SDBs requiring court application (Art. 3.7)."),
        "PI_LEDGER: Unclaimed Instruments for Internal Ledger": (detect_unclaimed_payment_instruments_ledger,
                                                                 "Detects unclaimed payment instruments for internal ledger (Art. 3.6)."),
        "CLAIMS: Claims Processing Pending (>1 Month)": (detect_claim_processing_pending,
                                                         "Detects customer claims (>1 month old) pending processing (Art. 4)."),
        "ANNUAL_RPT: Annual CBUAE Report Summary": (generate_annual_cbuae_report_summary,
                                                    "Generates summary for CBUAE Annual Report (Art. 3.10)."),
        "RETENTION: Record Retention Compliance": (check_record_retention_compliance,
                                                   "Checks record retention compliance (Art. 3.9 related).")
    }
    agent_func_tuple = agent_functions_compliance.get(selected_agent_key)
    if not agent_func_tuple:
        st.error(f"No function mapped for '{selected_agent_key}'. Please check UI configuration.")
        return

    agent_func, default_desc = agent_func_tuple
    data_filtered = pd.DataFrame()
    count = 0
    agent_run_desc = default_desc
    compliant_df = pd.DataFrame()  # For retention check

    with st.spinner(f"Running {selected_agent_key}..."):
        try:
            # Call the specific agent function with appropriate arguments
            if selected_agent_key == "FLAG: Flag Candidates (Not Yet Flagged)":
                data_filtered, count, agent_run_desc = agent_func(df.copy(), three_years_ago)
            elif selected_agent_key == "FREEZE: Statement Freeze Needed (Art. 7.3)":
                data_filtered, count, agent_run_desc = agent_func(df.copy(), three_years_ago)
            elif selected_agent_key == "RETENTION: Record Retention Compliance":
                data_filtered, compliant_df, agent_run_desc = agent_func(
                    df.copy())  # data_filtered is 'not_compliant_policy'
                count = len(data_filtered)  # Count for "not compliant" part
            else:  # Most functions just take df
                data_filtered, count, agent_run_desc = agent_func(df.copy())
            st.toast(f"{selected_agent_key} analysis complete!", icon="🔬")
        except Exception as e:
            st.error(f"Error running {selected_agent_key}: {e}")
            st.exception(e)
            return

    if selected_agent_key == "RETENTION: Record Retention Compliance":
        st.metric(f"Items Potentially Non-Compliant (Policy) by {selected_agent_key}", count)
        st.metric(f"Items Compliant/CBUAE Perpetual by {selected_agent_key}", len(compliant_df))
    else:
        st.metric(f"Items Identified by {selected_agent_key}", count)
    st.caption(agent_run_desc)

    if not data_filtered.empty:
        st.markdown(
            f"**Top {min(5, len(data_filtered))} items for '{selected_agent_key.split(': ')[1]}':**")  # Use cleaner name
        st.dataframe(data_filtered.head(min(5, len(data_filtered))), height=200, use_container_width=True)
        download_csv_button(data_filtered,
                            f"{selected_agent_key.replace(': ', '_').replace(' ', '_').lower()}_data.csv")

        if selected_agent_key == "RETENTION: Record Retention Compliance" and not compliant_df.empty:
            with st.expander("View Compliant / CBUAE Perpetual Records for Retention Check"):
                st.dataframe(compliant_df.head(min(5, len(compliant_df))), height=200, use_container_width=True)
                download_csv_button(compliant_df, f"retention_compliant_or_cb_data.csv")

        # Special action for FLAG agent: Logging
        if selected_agent_key == "FLAG: Flag Candidates (Not Yet Flagged)" and count > 0:
            if st.button("Log Flagging Instructions to DB", key="log_flags_btn_compliance"):
                threshold_days_for_log = (datetime.now() - three_years_ago).days
                status, msg = log_flag_instructions(data_filtered, agent_name_input, threshold_days_for_log)
                if status:
                    st.success(msg)
                else:
                    st.error(msg)

        # --- AI Insights for Individual Agent Data ---
        st.subheader(f"🤖 AI Insights for {selected_agent_key}")
        if llm:
            sample_for_ai = data_filtered.sample(min(len(data_filtered), 10)).to_csv(index=False)
            if st.button(f"Generate AI Insights for {selected_agent_key}", key=f"ai_btn_comp_{selected_agent_key}"):
                with st.spinner("Generating AI insights..."):
                    obs_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                    obs_chain = obs_prompt | llm | StrOutputParser()
                    observations = obs_chain.invoke({"data": sample_for_ai})
                    st.session_state[f"comp_obs_{selected_agent_key}"] = observations

                    trend_prompt = PromptTemplate.from_template(TREND_PROMPT)
                    trend_chain = trend_prompt | llm | StrOutputParser()
                    trends = trend_chain.invoke({"data": sample_for_ai})
                    st.session_state[f"comp_trend_{selected_agent_key}"] = trends

                    narr_prompt = PromptTemplate.from_template(NARRATION_PROMPT)
                    narr_chain = narr_prompt | llm | StrOutputParser()
                    narration = narr_chain.invoke({"observation": observations, "trend": trends})
                    st.session_state[f"comp_narr_{selected_agent_key}"] = narration

                    act_prompt = PromptTemplate.from_template(ACTION_PROMPT)
                    act_chain = act_prompt | llm | StrOutputParser()
                    actions = act_chain.invoke({"observation": observations, "trend": trends})
                    st.session_state[f"comp_act_{selected_agent_key}"] = actions
                st.toast("AI insights generated!", icon="💡")

            if f"comp_narr_{selected_agent_key}" in st.session_state:
                with st.expander("🔍 AI Observations", expanded=False):
                    st.markdown(st.session_state[f"comp_obs_{selected_agent_key}"])
                with st.expander("📈 AI Trend Analysis", expanded=False):
                    st.markdown(st.session_state[f"comp_trend_{selected_agent_key}"])
                with st.expander("📝 AI Narrative Summary", expanded=True):
                    st.markdown(st.session_state[f"comp_narr_{selected_agent_key}"])
                with st.expander("🚀 AI Recommended Actions", expanded=True):
                    st.markdown(st.session_state[f"comp_act_{selected_agent_key}"])

                individual_report_sections = [
                    {"title": f"Analysis Overview: {selected_agent_key}",
                     "content": f"{count} items identified. Description: {agent_run_desc}"},
                    {"title": "AI Observations", "content": st.session_state[f"comp_obs_{selected_agent_key}"]},
                    {"title": "AI Trend Analysis", "content": st.session_state[f"comp_trend_{selected_agent_key}"]},
                    {"title": "AI Narrative Summary", "content": st.session_state[f"comp_narr_{selected_agent_key}"]},
                    {"title": "AI Recommended Actions", "content": st.session_state[f"comp_act_{selected_agent_key}"]},
                ]
                download_pdf_button(f"{selected_agent_key}_Report", individual_report_sections,
                                    f"{selected_agent_key.replace(': ', '_').replace(' ', '_').lower()}_report.pdf")
        else:
            st.info("LLM not available for generating AI insights for individual checks.")
    elif not (
            selected_agent_key == "RETENTION: Record Retention Compliance" and not compliant_df.empty):  # Avoid "No items" if compliant_df has data
        st.info(f"No items identified by {selected_agent_key} for this specific category (e.g., 'not compliant').")