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
    st.header("Dormant Account Identification (CBUAE)")

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
    """
    Enhanced summarized view that includes comprehensive details from all dormant agents
    """
    st.subheader("📈 Comprehensive Dormancy Identification Summary")

    if st.button("🚀 Run Complete Dormancy Analysis", key="run_summary_dormant_analysis_button"):
        with st.spinner("Running all dormancy identification checks..."):
            results = run_all_dormant_identification_checks(
                df.copy(),
                report_date_str=report_date_str,
                dormant_flags_history_df=dormant_flags_history_df
            )
        st.session_state.dormant_summary_results_ui = results
        st.toast("Complete dormancy analysis finished!", icon="✅")

    if 'dormant_summary_results_ui' in st.session_state:
        results = st.session_state.dormant_summary_results_ui
        summary_kpis = results.get("summary_kpis", {})

        # Header Information
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Report Date", results.get('report_date_used', 'N/A'))
        with col2:
            st.metric("🏦 Total Accounts", results.get('total_accounts_analyzed', 'N/A'))
        with col3:
            st.metric("🚩 Total Flagged Dormant", summary_kpis.get("total_accounts_flagged_dormant", 0))

        # Key Performance Indicators
        st.subheader("🎯 Key Performance Indicators")
        kpi_cols = st.columns(4)

        with kpi_cols[0]:
            st.metric(
                "Dormancy Rate",
                f"{summary_kpis.get('percentage_dormant_of_total', 0):.2f}%",
                help="Percentage of total accounts flagged as dormant"
            )

        with kpi_cols[1]:
            balance_value = summary_kpis.get('total_dormant_balance_aed', 0)
            if isinstance(balance_value, (int, float)):
                st.metric("Total Dormant Balance", f"AED {balance_value:,.0f}")
            else:
                st.metric("Total Dormant Balance", str(balance_value))

        with kpi_cols[2]:
            st.metric(
                "High-Value Dormant",
                summary_kpis.get("count_high_value_dormant", 0),
                help="Accounts with balance ≥ AED 25,000"
            )

        with kpi_cols[3]:
            st.metric(
                "CB Transfer Eligible",
                summary_kpis.get("count_eligible_for_cb_transfer", 0),
                help="Accounts eligible for Central Bank transfer"
            )

        # Detailed Agent Results
        st.subheader("🔍 Detailed Analysis by Dormancy Category")

        # Create agent summary data
        agent_summary = {
            "Safe Deposit Boxes (Art 2.6)": {
                "count": results["sdb_dormant"]["count"],
                "desc": results["sdb_dormant"]["desc"],
                "details": results["sdb_dormant"]["details"],
                "icon": "🏦",
                "color": "red" if results["sdb_dormant"]["count"] > 0 else "gray"
            },
            "Investment Accounts (Art 2.3)": {
                "count": results["investment_dormant"]["count"],
                "desc": results["investment_dormant"]["desc"],
                "details": results["investment_dormant"]["details"],
                "icon": "📈",
                "color": "orange" if results["investment_dormant"]["count"] > 0 else "gray"
            },
            "Fixed Deposits (Art 2.2)": {
                "count": results["fixed_deposit_dormant"]["count"],
                "desc": results["fixed_deposit_dormant"]["desc"],
                "details": results["fixed_deposit_dormant"]["details"],
                "icon": "💰",
                "color": "yellow" if results["fixed_deposit_dormant"]["count"] > 0 else "gray"
            },
            "Demand Deposits (Art 2.1.1)": {
                "count": results["demand_deposit_dormant"]["count"],
                "desc": results["demand_deposit_dormant"]["desc"],
                "details": results["demand_deposit_dormant"]["details"],
                "icon": "💳",
                "color": "blue" if results["demand_deposit_dormant"]["count"] > 0 else "gray"
            },
            "Unclaimed Payment Instruments (Art 2.4)": {
                "count": results["unclaimed_instruments"]["count"],
                "desc": results["unclaimed_instruments"]["desc"],
                "details": results["unclaimed_instruments"]["details"],
                "icon": "📄",
                "color": "purple" if results["unclaimed_instruments"]["count"] > 0 else "gray"
            }
        }

        # Display agent results in expandable sections
        for agent_name, agent_data in agent_summary.items():
            with st.expander(f"{agent_data['icon']} {agent_name} - {agent_data['count']} items",
                             expanded=agent_data['count'] > 0):

                col_desc, col_metric = st.columns([3, 1])
                with col_desc:
                    if not agent_data['desc'].startswith("(Skipped"):
                        st.write(agent_data['desc'])
                    else:
                        st.warning(agent_data['desc'])

                with col_metric:
                    st.metric("Count", agent_data['count'])

                # Show details if available (excluding sample accounts)
                if agent_data['details'] and agent_data['count'] > 0:
                    st.markdown("**Additional Details:**")
                    # Filter out sample accounts from details
                    filtered_details = {k: v for k, v in agent_data['details'].items()
                                        if 'sample' not in k.lower()}

                    if filtered_details:
                        detail_cols = st.columns(len(filtered_details))
                        for i, (key, value) in enumerate(filtered_details.items()):
                            if i < len(detail_cols):
                                with detail_cols[i]:
                                    if isinstance(value, (int, float)):
                                        if 'amount' in key.lower() or 'balance' in key.lower():
                                            st.metric(key.replace('_', ' ').title(), f"AED {value:,.0f}")
                                        else:
                                            st.metric(key.replace('_', ' ').title(), f"{value:,.0f}")
                                    else:
                                        st.metric(key.replace('_', ' ').title(), str(value))

        # Process & Action Items
        st.subheader("⚡ Process & Action Items")

        action_cols = st.columns(3)

        with action_cols[0]:
            st.metric(
                "🔄 Article 3 Process Needed",
                summary_kpis.get("count_needing_art3_process", 0),
                help="Accounts requiring contact/wait period process"
            )
            if summary_kpis.get("count_needing_art3_process", 0) > 0:
                art3_details = results["art3_process_needed"]["details"]
                st.caption(f"• Needs initial contact: {art3_details.get('needs_initial_contact', 0)}")
                st.caption(f"• In 3-month wait: {art3_details.get('in_3_month_wait_period', 0)}")

        with action_cols[1]:
            st.metric(
                "📞 Proactive Contact Needed",
                summary_kpis.get("count_needing_proactive_contact", 0),
                help="Accounts nearing dormancy requiring preventive contact"
            )

        with action_cols[2]:
            st.metric(
                "🔄 Dormant-to-Active Transitions",
                summary_kpis.get("count_dormant_to_active_transitions", 0),
                help="Previously dormant accounts showing recent activity"
            )

        # Visual Charts with Pie Charts
        st.subheader("📊 Visual Analytics")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Dormancy by Category**")
            dormancy_data = {
                "Safe Deposit": summary_kpis.get("count_sdb_dormant", 0),
                "Investment": summary_kpis.get("count_investment_dormant", 0),
                "Fixed Deposit": summary_kpis.get("count_fixed_deposit_dormant", 0),
                "Demand Deposit": summary_kpis.get("count_demand_deposit_dormant", 0),
                "Unclaimed Instruments": summary_kpis.get("count_unclaimed_instruments", 0),
            }

            chart_df = pd.DataFrame(list(dormancy_data.items()), columns=['Category', 'Count'])
            chart_df = chart_df[chart_df['Count'] > 0]  # Only show categories with data

            if not chart_df.empty:
                try:
                    import plotly.express as px
                    fig1 = px.pie(chart_df, values='Count', names='Category',
                                  title='Distribution of Dormant Accounts by Category',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(height=400, showlegend=True, legend=dict(orientation="v", x=1.05))
                    st.plotly_chart(fig1, use_container_width=True)
                except ImportError:
                    # Fallback to bar chart if plotly not available
                    st.bar_chart(chart_df.set_index('Category'), height=300)
            else:
                st.info("No dormant accounts identified across categories.")

        with chart_col2:
            st.markdown("**Process Status Overview**")
            process_data = {
                "CB Transfer Eligible": summary_kpis.get("count_eligible_for_cb_transfer", 0),
                "Article 3 Process": summary_kpis.get("count_needing_art3_process", 0),
                "Proactive Contact": summary_kpis.get("count_needing_proactive_contact", 0),
                "High Value (≥25k)": summary_kpis.get("count_high_value_dormant", 0),
                "Reactivated": summary_kpis.get("count_dormant_to_active_transitions", 0),
            }

            process_df = pd.DataFrame(list(process_data.items()), columns=['Status', 'Count'])
            process_df = process_df[process_df['Count'] > 0]

            if not process_df.empty:
                try:
                    import plotly.express as px
                    fig2 = px.pie(process_df, values='Count', names='Status',
                                  title='Distribution of Process Status Items',
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig2.update_traces(textposition='inside', textinfo='percent+label')
                    fig2.update_layout(height=400, showlegend=True, legend=dict(orientation="v", x=1.05))
                    st.plotly_chart(fig2, use_container_width=True)
                except ImportError:
                    # Fallback to bar chart if plotly not available
                    st.bar_chart(process_df.set_index('Status'), height=300)
            else:
                st.info("No process-specific items identified.")

        # Prepare comprehensive text for AI analysis
        # Format balance values properly
        total_balance = summary_kpis.get('total_dormant_balance_aed', 0)
        total_balance_formatted = f"AED {total_balance:,.0f}" if isinstance(total_balance, (int, float)) else str(
            total_balance)

        high_value_balance = summary_kpis.get('total_high_value_dormant_balance_aed', 0)
        high_value_balance_formatted = f"AED {high_value_balance:,.0f}" if isinstance(high_value_balance,
                                                                                      (int, float)) else str(
            high_value_balance)

        comprehensive_summary = f"""
DORMANCY ANALYSIS COMPREHENSIVE REPORT
=====================================
Report Date: {results.get('report_date_used')}
Total Accounts Analyzed: {results.get('total_accounts_analyzed')}

EXECUTIVE SUMMARY:
- Total Dormant Accounts: {summary_kpis.get('total_accounts_flagged_dormant', 0)} ({summary_kpis.get('percentage_dormant_of_total', 0):.2f}%)
- Total Dormant Balance: {total_balance_formatted}
- High-Value Dormant: {summary_kpis.get('count_high_value_dormant', 0)} accounts ({high_value_balance_formatted})

DETAILED FINDINGS BY CATEGORY:
"""

        for category, data in agent_summary.items():
            if not data['desc'].startswith("(Skipped"):
                comprehensive_summary += f"\n{category}:\n- {data['desc']}\n"
                if data['details']:
                    # Filter out sample accounts from comprehensive summary
                    filtered_details = {k: v for k, v in data['details'].items()
                                        if 'sample' not in k.lower()}
                    for key, value in filtered_details.items():
                        comprehensive_summary += f"  - {key.replace('_', ' ').title()}: {value}\n"

        comprehensive_summary += f"""
PROCESS & ACTION ITEMS:
- Article 3 Process Required: {summary_kpis.get('count_needing_art3_process', 0)} accounts
- Proactive Contact Needed: {summary_kpis.get('count_needing_proactive_contact', 0)} accounts  
- Central Bank Transfer Eligible: {summary_kpis.get('count_eligible_for_cb_transfer', 0)} items
- Dormant-to-Active Transitions: {summary_kpis.get('count_dormant_to_active_transitions', 0)} accounts

FINANCIAL IMPACT:
- Total Unclaimed Instruments Value: AED {summary_kpis.get('total_unclaimed_instruments_value', 0):,.0f}
- High-Value Dormant Balance: {high_value_balance_formatted}
"""

        # AI-Generated Executive Summary
        st.subheader("🤖 AI-Generated Executive Summary & Strategic Insights")

        if llm:
            try:
                with st.spinner("Generating comprehensive AI analysis..."):
                    prompt_template = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
                    chain = prompt_template | llm | StrOutputParser()
                    ai_summary = chain.invoke({"analysis_details": comprehensive_summary})

                st.markdown(ai_summary)
                st.session_state.dormant_ai_summary_text_ui = ai_summary

                # Save to database if function exists
                try:
                    save_summary_to_db("dormant_analysis", ai_summary, results.get('report_date_used'))
                    st.success("Analysis saved to database!")
                except:
                    pass  # Silently continue if save function not available

            except Exception as e:
                st.error(f"AI summary generation failed: {e}")
                fallback_summary = get_fallback_response("dormant_summary")
                st.session_state.dormant_ai_summary_text_ui = f"{fallback_summary}\n\n{comprehensive_summary}"
                st.warning("Using fallback summary due to AI service unavailability.")
                st.text_area("Detailed Analysis", comprehensive_summary, height=300)
        else:
            st.warning("LLM not available. Displaying comprehensive raw analysis.")
            st.text_area("Comprehensive Analysis Details", comprehensive_summary, height=400)
            st.session_state.dormant_ai_summary_text_ui = comprehensive_summary

        # Export Options
        st.subheader("📥 Export Comprehensive Report")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            # PDF Export
            comprehensive_report_sections = [
                {"title": "Executive Summary",
                 "content": f"Total Accounts: {results.get('total_accounts_analyzed')}\nDormant Accounts: {summary_kpis.get('total_accounts_flagged_dormant', 0)}\nDormancy Rate: {summary_kpis.get('percentage_dormant_of_total', 0):.2f}%"},
                {"title": "Detailed Analysis", "content": comprehensive_summary},
                {"title": "AI Strategic Insights",
                 "content": st.session_state.get("dormant_ai_summary_text_ui", "AI Summary not available.")},
                {"title": "Raw Data Summary", "content": str(summary_kpis)}
            ]
            download_pdf_button(
                "Comprehensive_Dormancy_Analysis_Report",
                comprehensive_report_sections,
                "comprehensive_dormancy_report.pdf"
            )

        with export_col2:
            # CSV Export of all identified dormant accounts
            if summary_kpis.get("total_accounts_flagged_dormant", 0) > 0:
                all_dormant_accounts = df[
                    df.get('Expected_Account_Dormant', pd.Series(dtype=str)).astype(str).str.lower().isin(
                        ['yes', 'true', '1'])]
                if not all_dormant_accounts.empty:
                    download_csv_button(
                        all_dormant_accounts,
                        "all_dormant_accounts_identified.csv"
                    )

        # Raw Data Explorer
        with st.expander("🔍 Raw Results Explorer (Technical Details)", expanded=False):
            result_category = st.selectbox(
                "Select Category to Explore:",
                ["summary_kpis", "sdb_dormant", "investment_dormant", "fixed_deposit_dormant",
                 "demand_deposit_dormant", "unclaimed_instruments", "eligible_for_cb_transfer",
                 "art3_process_needed", "proactive_contact_needed", "high_value_dormant", "dormant_to_active"]
            )

            if result_category in results:
                st.json(results[result_category], expanded=True)
            else:
                st.error(f"Category '{result_category}' not found in results.")

        # Alerts and Recommendations
        st.subheader("⚠️ Key Alerts & Immediate Actions Required")

        alerts = []
        if summary_kpis.get("count_high_value_dormant", 0) > 0:
            alerts.append(
                f"🚨 **HIGH PRIORITY**: {summary_kpis.get('count_high_value_dormant', 0)} high-value dormant accounts identified (≥AED 25,000)")

        if summary_kpis.get("count_eligible_for_cb_transfer", 0) > 0:
            alerts.append(
                f"📋 **REGULATORY**: {summary_kpis.get('count_eligible_for_cb_transfer', 0)} accounts/instruments eligible for CBUAE transfer")

        if summary_kpis.get("count_needing_art3_process", 0) > 0:
            alerts.append(
                f"📞 **ACTION REQUIRED**: {summary_kpis.get('count_needing_art3_process', 0)} accounts need Article 3 process (customer contact)")

        if summary_kpis.get("count_needing_proactive_contact", 0) > 0:
            alerts.append(
                f"🔔 **PREVENTIVE**: {summary_kpis.get('count_needing_proactive_contact', 0)} accounts nearing dormancy - proactive contact recommended")

        if alerts:
            for alert in alerts:
                st.markdown(alert)
        else:
            st.success("✅ No immediate alerts. All dormancy processes appear to be under control.")

    else:
        st.info(
            "Click 'Run Complete Dormancy Analysis' to generate comprehensive dormancy insights across all categories.")
        st.markdown("""
        **This comprehensive analysis will provide:**
        - 📊 Complete overview of all dormancy categories (Arts 2.1-2.6)
        - 🎯 Key performance indicators and metrics
        - 📈 Visual analytics with interactive pie charts
        - ⚡ Process status and action items
        - 🤖 AI-generated strategic insights
        - 📥 Exportable reports (PDF/CSV)
        - ⚠️ Priority alerts and recommendations
        """)

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