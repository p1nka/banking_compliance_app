import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
# Assuming your backend dormant.py is in an 'agents' directory
from agents.dormant import (
    run_all_dormant_identification_checks, # Renamed from run_all_dormant_checks
    check_safe_deposit_dormancy,           # Renamed from check_safe_deposit
    check_investment_inactivity,
    check_fixed_deposit_inactivity,
    check_demand_deposit_inactivity,
    check_unclaimed_payment_instruments,   # Renamed from check_bankers_cheques
    check_eligible_for_cb_transfer,    # Renamed from check_transfer_to_central_bank
    check_art3_process_needed,           # Renamed from check_art3_process_required
    check_contact_attempts_needed,
    check_high_value_dormant_accounts,   # Renamed from check_high_value_dormant
    check_dormant_to_active_transitions # Renamed from check_dormant_to_active
)
# Assuming these utility modules exist as per your original code
from database.pipeline import AgentDatabasePipeline # If needed for dormant_flags_history
from data.exporters import download_pdf_button, download_csv_button
from ai.llm import (
    get_fallback_response,
    DORMANT_SUMMARY_PROMPT,
    OBSERVATION_PROMPT,
    TREND_PROMPT,
    NARRATION_PROMPT,
    ACTION_PROMPT
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from database.operations import save_summary_to_db # If you intend to save insights


# --- Main Rendering Function for Dormant UI ---
def render_dormant_analyzer(df, report_date_str, llm, dormant_flags_history_df):
    """
    Main function to render the Dormant Account Analyzer UI.
    """
    st.header("🇦🇪 Dormant Account Identification (CBUAE)")

    agent_options_dormant = [
        "📊 Summarized Dormancy Analysis (All Checks)",
        "--- Individual Agent Checks ---",
        "SDB: Safe Deposit Box Dormancy",
        "INV: Investment Account Inactivity",
        "FD: Fixed Deposit Inactivity",
        "DD: Demand Deposit Inactivity",
        "PI: Unclaimed Payment Instruments",
        "CB: Eligible for CBUAE Transfer",
        "ART3: Article 3 Process Needed",
        "CON: Contact Attempts Needed (Proactive)",
        "HV: High-Value Dormant Accounts (≥ AED 25k)",
        "DA: Dormant-to-Active Transitions"
    ]

    selected_agent_dormant = st.selectbox(
        "Select Dormancy Identification Task",
        agent_options_dormant,
        key="dormant_agent_selector_ui"
    )

    if selected_agent_dormant == "📊 Summarized Dormancy Analysis (All Checks)":
        render_summarized_dormant_analysis_view(df, report_date_str, llm, dormant_flags_history_df)
    elif selected_agent_dormant != "--- Individual Agent Checks ---":
        render_individual_dormant_agent_view(df, selected_agent_dormant, report_date_str, llm, dormant_flags_history_df)

# --- Summarized View ---
def render_summarized_dormant_analysis_view(df, report_date_str, llm, dormant_flags_history_df):
    st.subheader("📈 Summarized Dormancy Identification Results")

    if st.button("🚀 Run Summarized Dormancy Analysis", key="run_summary_dormant_analysis_button"):
        with st.spinner("Running all dormancy identification checks..."):
            results = run_all_dormant_identification_checks(
                df.copy(),
                report_date_str=report_date_str,
                dormant_flags_history_df=dormant_flags_history_df
            )
        st.session_state.dormant_summary_results_ui = results
        st.toast("Summarized dormancy analysis complete!", icon="✅")

    if 'dormant_summary_results_ui' in st.session_state:
        results = st.session_state.dormant_summary_results_ui
        summary_kpis = results.get("summary_kpis", {})

        st.markdown(f"**Report Date Used:** `{results.get('report_date_used', 'N/A')}` | **Total Accounts Analyzed:** `{results.get('total_accounts_analyzed', 'N/A')}`")

        st.subheader("Key Performance Indicators (KPIs)")
        cols_kpi = st.columns(3)
        cols_kpi[0].metric("Total Accounts Flagged Dormant", summary_kpis.get("total_accounts_flagged_dormant", 0))
        cols_kpi[1].metric("% Dormant of Total", f"{summary_kpis.get('percentage_dormant_of_total', 0):.2f}%")
        cols_kpi[2].metric("Total Dormant Balance (AED)", f"{summary_kpis.get('total_dormant_balance_aed', 0):,.2f}" if isinstance(summary_kpis.get('total_dormant_balance_aed'), (int, float)) else summary_kpis.get('total_dormant_balance_aed', "N/A"))

        # --- Charts for Summary ---
        st.subheader("Visual Insights")
        # Chart 1: Dormancy Categories Count
        dormancy_counts_data = {
            "SDB": summary_kpis.get("count_sdb_dormant", 0),
            "Investment": summary_kpis.get("count_investment_dormant", 0),
            "Fixed Deposit": summary_kpis.get("count_fixed_deposit_dormant", 0),
            "Demand Deposit": summary_kpis.get("count_demand_deposit_dormant", 0),
            "Unclaimed PI": summary_kpis.get("count_unclaimed_instruments", 0),
        }
        dormancy_counts_df = pd.DataFrame(list(dormancy_counts_data.items()), columns=['Category', 'Count']).set_index('Category')
        if not dormancy_counts_df.empty:
            st.bar_chart(dormancy_counts_df, height=300)
        else:
            st.info("No data for dormancy category chart.")

        # Prepare text for AI summary
        summary_input_text = f"Dormancy Analysis Report (Date: {results.get('report_date_used')}, Total Analyzed: {results.get('total_accounts_analyzed')})\n"
        for key, val_dict in results.items():
            if isinstance(val_dict, dict) and "desc" in val_dict and "count" in val_dict:
                 if not val_dict["desc"].startswith("(Skipped"): # Avoid adding skipped checks to AI summary
                    summary_input_text += f"- {val_dict['desc']}\n"
        summary_input_text += "\nSummary KPIs:\n"
        for k,v in summary_kpis.items():
            summary_input_text += f"  - {k.replace('_', ' ').title()}: {v}\n"


        st.subheader("📝 AI Generated Summary & Insights")
        if llm:
            try:
                with st.spinner("Generating AI summary for all dormancy checks..."):
                    prompt_template = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
                    chain = prompt_template | llm | StrOutputParser()
                    ai_summary = chain.invoke({"analysis_details": summary_input_text})
                st.markdown(ai_summary)
                st.session_state.dormant_ai_summary_text_ui = ai_summary
            except Exception as e:
                st.error(f"AI summary generation failed: {e}")
                st.session_state.dormant_ai_summary_text_ui = get_fallback_response("dormant_summary") + f"\n\nRaw Data:\n{summary_input_text}"
                st.warning(st.session_state.dormant_ai_summary_text_ui)
        else:
            st.warning("LLM not available. Displaying raw findings.")
            st.text_area("Raw Findings for Summary", summary_input_text, height=200)
            st.session_state.dormant_ai_summary_text_ui = summary_input_text

        # --- Export Options for Summary ---
        st.subheader("⬇️ Export Summarized Report")
        summary_report_sections = [
            {"title": "Dormancy Analysis Overview", "content": summary_input_text},
            {"title": "AI Generated Summary", "content": st.session_state.get("dormant_ai_summary_text_ui", "AI Summary not generated.")}
        ]
        download_pdf_button("Dormancy_Analysis_Summary_Report", summary_report_sections, "dormancy_summary_report.pdf")

        with st.expander("View Raw Results for All Dormancy Checks"):
            st.json(results, expanded=False)


# --- Individual Agent View ---
def render_individual_dormant_agent_view(df, selected_agent_key, report_date_str, llm, dormant_flags_history_df):
    st.subheader(f"🔍 Results for: {selected_agent_key}")

    # Agent mapping (key from dropdown to function and description)
    # Ensure function names match your `agents.dormant.py`
    agent_functions_dormant = {
        "SDB: Safe Deposit Box Dormancy": (check_safe_deposit_dormancy, "Identifies Safe Deposit Boxes meeting dormancy criteria (Art. 2.6)."),
        "INV: Investment Account Inactivity": (check_investment_inactivity, "Identifies Investment Accounts meeting dormancy criteria (Art. 2.3)."),
        "FD: Fixed Deposit Inactivity": (check_fixed_deposit_inactivity, "Identifies Fixed/Term Deposit accounts meeting dormancy criteria (Art. 2.2)."),
        "DD: Demand Deposit Inactivity": (check_demand_deposit_inactivity, "Identifies Demand Deposit accounts meeting dormancy criteria (Art. 2.1.1)."),
        "PI: Unclaimed Payment Instruments": (check_unclaimed_payment_instruments, "Identifies unclaimed Bankers Cheques, Bank Drafts, Cashier Orders (Art. 2.4)."),
        "CB: Eligible for CBUAE Transfer": (check_eligible_for_cb_transfer, "Identifies accounts/balances eligible for transfer to Central Bank (Art. 8.1, 8.2)."),
        "ART3: Article 3 Process Needed": (check_art3_process_needed, "Identifies accounts needing/undergoing Art. 3 process (contact/wait period)."),
        "CON: Contact Attempts Needed (Proactive)": (check_contact_attempts_needed, "Identifies accounts nearing dormancy needing proactive contact attempts."),
        "HV: High-Value Dormant Accounts (≥ AED 25k)": (check_high_value_dormant_accounts, "Identifies high-value dormant accounts (Balance >= AED 25,000)."),
        "DA: Dormant-to-Active Transitions": (check_dormant_to_active_transitions, "Identifies accounts that were previously dormant but have shown recent activity.")
    }

    agent_func_tuple = agent_functions_dormant.get(selected_agent_key)
    if not agent_func_tuple:
        st.error(f"No function mapped for '{selected_agent_key}'. Please check UI configuration.")
        return

    agent_func, default_desc = agent_func_tuple
    data_filtered = pd.DataFrame()
    count = 0
    agent_run_desc = default_desc
    details = {}
    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")


    with st.spinner(f"Running {selected_agent_key}..."):
        try:
            # Call the specific agent function
            if selected_agent_key == "HV: High-Value Dormant Accounts (≥ AED 25k)":
                data_filtered, count, agent_run_desc, details = agent_func(df.copy())
            elif selected_agent_key == "DA: Dormant-to-Active Transitions":
                 data_filtered, count, agent_run_desc, details = agent_func(df.copy(), report_date, dormant_flags_history_df)
            else: # Most other functions take df and report_date
                data_filtered, count, agent_run_desc, details = agent_func(df.copy(), report_date)
            st.toast(f"{selected_agent_key} analysis complete!", icon="🔬")
        except Exception as e:
            st.error(f"Error running {selected_agent_key}: {e}")
            st.exception(e)
            return

    st.metric(f"Items Identified by {selected_agent_key}", count)
    st.caption(agent_run_desc)

    if details:
        with st.expander("Additional Details from Check"):
            st.json(details, expanded=False)

    if not data_filtered.empty:
        st.markdown(f"**Top {min(5, len(data_filtered))} items identified:**")
        st.dataframe(data_filtered.head(min(5, len(data_filtered))), height=200, use_container_width=True)
        download_csv_button(data_filtered, f"{selected_agent_key.replace(': ', '_').replace(' ', '_').lower()}_data.csv")

        # --- AI Insights for Individual Agent Data ---
        st.subheader(f"🤖 AI Insights for {selected_agent_key}")
        if llm:
            sample_for_ai = data_filtered.sample(min(len(data_filtered), 10)).to_csv(index=False) # Sample up to 10 rows
            if st.button(f"Generate AI Insights for {selected_agent_key}", key=f"ai_btn_{selected_agent_key}"):
                with st.spinner("Generating AI insights..."):
                    # Observations
                    obs_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                    obs_chain = obs_prompt | llm | StrOutputParser()
                    observations = obs_chain.invoke({"data": sample_for_ai})
                    st.session_state[f"dormant_obs_{selected_agent_key}"] = observations

                    # Trends
                    trend_prompt = PromptTemplate.from_template(TREND_PROMPT)
                    trend_chain = trend_prompt | llm | StrOutputParser()
                    trends = trend_chain.invoke({"data": sample_for_ai})
                    st.session_state[f"dormant_trend_{selected_agent_key}"] = trends

                    # Narration
                    narr_prompt = PromptTemplate.from_template(NARRATION_PROMPT)
                    narr_chain = narr_prompt | llm | StrOutputParser()
                    narration = narr_chain.invoke({"observation": observations, "trend": trends})
                    st.session_state[f"dormant_narr_{selected_agent_key}"] = narration

                    # Actions
                    act_prompt = PromptTemplate.from_template(ACTION_PROMPT)
                    act_chain = act_prompt | llm | StrOutputParser()
                    actions = act_chain.invoke({"observation": observations, "trend": trends})
                    st.session_state[f"dormant_act_{selected_agent_key}"] = actions
                st.toast("AI insights generated!", icon="💡")

            if f"dormant_narr_{selected_agent_key}" in st.session_state:
                with st.expander("🔍 AI Observations", expanded=False):
                    st.markdown(st.session_state[f"dormant_obs_{selected_agent_key}"])
                with st.expander("📈 AI Trend Analysis", expanded=False):
                    st.markdown(st.session_state[f"dormant_trend_{selected_agent_key}"])
                with st.expander("📝 AI Narrative Summary", expanded=True):
                    st.markdown(st.session_state[f"dormant_narr_{selected_agent_key}"])
                with st.expander("🚀 AI Recommended Actions", expanded=True):
                    st.markdown(st.session_state[f"dormant_act_{selected_agent_key}"])

                # PDF Export for individual agent insight
                individual_report_sections = [
                    {"title": f"Analysis Overview: {selected_agent_key}", "content": f"{count} items identified. Description: {agent_run_desc}"},
                    {"title": "AI Observations", "content": st.session_state[f"dormant_obs_{selected_agent_key}"]},
                    {"title": "AI Trend Analysis", "content": st.session_state[f"dormant_trend_{selected_agent_key}"]},
                    {"title": "AI Narrative Summary", "content": st.session_state[f"dormant_narr_{selected_agent_key}"]},
                    {"title": "AI Recommended Actions", "content": st.session_state[f"dormant_act_{selected_agent_key}"]},
                ]
                download_pdf_button(f"{selected_agent_key}_Report", individual_report_sections, f"{selected_agent_key.replace(': ', '_').replace(' ', '_').lower()}_report.pdf")
        else:
            st.info("LLM not available for generating AI insights for individual checks.")
    else:
        st.info(f"No items identified by {selected_agent_key}.")