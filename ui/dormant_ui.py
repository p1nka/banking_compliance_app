import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import json
import logging

# Import the agent classes from the dormant.py file
from agents.dormant import (
    DormantAccountOrchestrator,
    SafeDepositBoxAgent,
    InvestmentAccountAgent,
    FixedDepositAgent,
    DemandDepositAgent,
    PaymentInstrumentsAgent,
    CBTransferAgent,
    Article3ProcessAgent,
    HighValueAccountAgent,
    TransitionDetectionAgent,
    ActivityStatus,
    AccountType,
    CustomerTier,
    Priority,
    RiskLevel
)

# Import utility modules - these should exist in your system
from database.pipelines import AgentDatabasePipeline
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
from database.operations import save_summary_to_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_dormant_agents(llm_client, config):
    """Initialize all dormant agents with the provided LLM client and config."""
    agents = {}
    agent_classes = DormantAccountOrchestrator.AGENT_CLASS_MAP

    for agent_name, agent_class in agent_classes.items():
        try:
            agents[agent_name] = agent_class(llm_client, config)
        except Exception as e:
            logger.error(f"Failed to initialize {agent_name}: {e}")
            agents[agent_name] = None

    return agents


def prepare_account_data(df_row, account_type=None):
    """Prepare account data from DataFrame row for agent processing."""
    account_data = {
        'account_id': df_row.get('Account_Number', 'Unknown'),
        'account_type': account_type or df_row.get('Account_Type', 'unknown'),
        'balance': float(df_row.get('Balance', 0) or 0),
        'last_activity_date': df_row.get('Last_Transaction_Date'),
        'maturity_date': df_row.get('Maturity_Date'),
        'customer_tier': df_row.get('Customer_Tier', 'standard'),
        'contact_attempts_made': int(df_row.get('Contact_Attempts', 0) or 0),
        'previous_dormancy_status': df_row.get('Previous_Status'),
        'current_activity_status': df_row.get('Current_Status', 'active'),
        'dormancy_status': df_row.get('Dormancy_Status', 'active')
    }

    # Handle date conversion
    for date_field in ['last_activity_date', 'maturity_date']:
        if account_data[date_field] and isinstance(account_data[date_field], str):
            try:
                account_data[date_field] = datetime.fromisoformat(account_data[date_field])
            except:
                try:
                    account_data[date_field] = datetime.strptime(account_data[date_field], "%Y-%m-%d")
                except:
                    account_data[date_field] = None

    return account_data


def run_single_agent_analysis(agent, df, account_type_filter=None):
    """Run a single agent analysis on the provided DataFrame."""
    results = []
    processed_count = 0

    if agent is None:
        return results, processed_count, "Agent not initialized"

    for idx, row in df.iterrows():
        if account_type_filter and row.get('Account_Type') != account_type_filter:
            continue

        account_data = prepare_account_data(row, account_type_filter)

        try:
            result = agent.execute(account_data)
            if result and result.get('status') not in [ActivityStatus.ACTIVE.value, 'N/A', 'No Transition',
                                                       'Standard Value']:
                result['account_data'] = account_data
                results.append(result)
            processed_count += 1
        except Exception as e:
            logger.error(
                f"Error processing account {account_data['account_id']} with agent {agent.__class__.__name__}: {e}")

    return results, processed_count, f"Processed {processed_count} accounts"


def run_all_dormant_identification_checks(df, report_date_str, llm_client, config, dormant_flags_history_df=None):
    """Run comprehensive dormancy analysis using all agents."""

    if llm_client is None or config is None:
        st.error("LLM client and config are required for dormancy analysis. Please ensure proper initialization.")
        return None

    # Initialize agents
    agents = initialize_dormant_agents(llm_client, config)

    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
    total_accounts = len(df)

    results = {
        'report_date_used': report_date_str,
        'total_accounts_analyzed': total_accounts,
        'summary_kpis': {
            'total_accounts_flagged_dormant': 0,
            'percentage_dormant_of_total': 0.0,
            'total_dormant_balance_aed': 0.0,
            'count_high_value_dormant': 0,
            'total_high_value_dormant_balance_aed': 0.0,
            'count_eligible_for_cb_transfer': 0,
            'count_sdb_dormant': 0,
            'count_investment_dormant': 0,
            'count_fixed_deposit_dormant': 0,
            'count_demand_deposit_dormant': 0,
            'count_unclaimed_instruments': 0,
            'count_needing_art3_process': 0,
            'count_needing_proactive_contact': 0,
            'count_dormant_to_active_transitions': 0,
            'total_unclaimed_instruments_value': 0.0
        }
    }

    # Safe Deposit Box Analysis
    sdb_results, sdb_count, sdb_desc = run_single_agent_analysis(
        agents.get('safe_deposit_box_agent'),
        df,
        AccountType.SAFE_DEPOSIT.value
    )
    results['sdb_dormant'] = {
        'count': len(sdb_results),
        'desc': f"Safe Deposit Boxes flagged as dormant (3+ years inactive): {len(sdb_results)}",
        'details': {'total_processed': sdb_count}
    }
    results['summary_kpis']['count_sdb_dormant'] = len(sdb_results)

    # Investment Account Analysis
    inv_results, inv_count, inv_desc = run_single_agent_analysis(
        agents.get('investment_account_agent'),
        df,
        AccountType.INVESTMENT.value
    )
    results['investment_dormant'] = {
        'count': len(inv_results),
        'desc': f"Investment accounts flagged as dormant (3+ years inactive): {len(inv_results)}",
        'details': {'total_processed': inv_count}
    }
    results['summary_kpis']['count_investment_dormant'] = len(inv_results)

    # Fixed Deposit Analysis
    fd_results, fd_count, fd_desc = run_single_agent_analysis(
        agents.get('fixed_deposit_agent'),
        df,
        AccountType.FIXED_DEPOSIT.value
    )
    results['fixed_deposit_dormant'] = {
        'count': len(fd_results),
        'desc': f"Fixed deposits unclaimed post-maturity (3+ years): {len(fd_results)}",
        'details': {'total_processed': fd_count}
    }
    results['summary_kpis']['count_fixed_deposit_dormant'] = len(fd_results)

    # Demand Deposit Analysis
    dd_results, dd_count, dd_desc = run_single_agent_analysis(
        agents.get('demand_deposit_agent'),
        df,
        AccountType.DEMAND_DEPOSIT.value
    )
    results['demand_deposit_dormant'] = {
        'count': len(dd_results),
        'desc': f"Demand deposits flagged as dormant (3+ years inactive): {len(dd_results)}",
        'details': {'total_processed': dd_count}
    }
    results['summary_kpis']['count_demand_deposit_dormant'] = len(dd_results)

    # Payment Instruments Analysis
    pi_results, pi_count, pi_desc = run_single_agent_analysis(
        agents.get('payment_instruments_agent'),
        df,
        AccountType.UNCLAIMED_INSTRUMENT.value
    )
    results['unclaimed_instruments'] = {
        'count': len(pi_results),
        'desc': f"Unclaimed payment instruments (1+ year): {len(pi_results)}",
        'details': {'total_processed': pi_count}
    }
    results['summary_kpis']['count_unclaimed_instruments'] = len(pi_results)

    # Central Bank Transfer Eligibility
    cb_results, cb_count, cb_desc = run_single_agent_analysis(
        agents.get('cb_transfer_agent'),
        df
    )
    eligible_cb_results = [r for r in cb_results if r.get('eligible', False)]
    results['eligible_for_cb_transfer'] = {
        'count': len(eligible_cb_results),
        'desc': f"Items eligible for CBUAE transfer (5+ years dormant): {len(eligible_cb_results)}",
        'details': {'total_processed': cb_count}
    }
    results['summary_kpis']['count_eligible_for_cb_transfer'] = len(eligible_cb_results)

    # Article 3 Process Analysis
    art3_results, art3_count, art3_desc = run_single_agent_analysis(
        agents.get('article_3_process_agent'),
        df
    )
    needing_art3 = [r for r in art3_results if r.get('status') == 'Process Pending']
    results['art3_process_needed'] = {
        'count': len(needing_art3),
        'desc': f"Accounts requiring Article 3 process (contact procedures): {len(needing_art3)}",
        'details': {
            'total_processed': art3_count,
            'needs_initial_contact': len(needing_art3),
            'in_3_month_wait_period': 0  # Would need additional logic to determine
        }
    }
    results['summary_kpis']['count_needing_art3_process'] = len(needing_art3)

    # High Value Account Analysis
    hv_results, hv_count, hv_desc = run_single_agent_analysis(
        agents.get('high_value_account_agent'),
        df
    )
    high_value_dormant = [r for r in hv_results if r.get('status') == 'High Value Dormant']
    total_hv_balance = sum(r.get('account_data', {}).get('balance', 0) for r in high_value_dormant)

    results['high_value_dormant'] = {
        'count': len(high_value_dormant),
        'desc': f"High-value dormant accounts (≥AED 100,000): {len(high_value_dormant)}",
        'details': {
            'total_processed': hv_count,
            'total_balance': total_hv_balance
        }
    }
    results['summary_kpis']['count_high_value_dormant'] = len(high_value_dormant)
    results['summary_kpis']['total_high_value_dormant_balance_aed'] = total_hv_balance

    # Transition Detection Analysis
    td_results, td_count, td_desc = run_single_agent_analysis(
        agents.get('transition_detection_agent'),
        df
    )
    reactivated = [r for r in td_results if r.get('status') == 'Reactivated']
    results['dormant_to_active'] = {
        'count': len(reactivated),
        'desc': f"Accounts reactivated (dormant to active): {len(reactivated)}",
        'details': {'total_processed': td_count}
    }
    results['summary_kpis']['count_dormant_to_active_transitions'] = len(reactivated)

    # Proactive Contact Analysis (placeholder - would need additional logic)
    results['proactive_contact_needed'] = {
        'count': 0,
        'desc': "Accounts nearing dormancy requiring proactive contact: 0",
        'details': {'total_processed': 0}
    }
    results['summary_kpis']['count_needing_proactive_contact'] = 0

    # Calculate overall KPIs
    all_dormant_results = sdb_results + inv_results + fd_results + dd_results + pi_results
    unique_dormant_accounts = len(set(r.get('account_data', {}).get('account_id') for r in all_dormant_results))
    total_dormant_balance = sum(r.get('account_data', {}).get('balance', 0) for r in all_dormant_results)

    results['summary_kpis']['total_accounts_flagged_dormant'] = unique_dormant_accounts
    results['summary_kpis']['percentage_dormant_of_total'] = (
                unique_dormant_accounts / total_accounts * 100) if total_accounts > 0 else 0
    results['summary_kpis']['total_dormant_balance_aed'] = total_dormant_balance
    results['summary_kpis']['total_unclaimed_instruments_value'] = sum(
        r.get('account_data', {}).get('balance', 0) for r in pi_results)

    return results


# --- Main Rendering Function for Dormant UI ---
def render_dormant_analyzer(df, report_date_str, llm, dormant_flags_history_df, config=None):
    """
    Main function to render the Dormant Account Analyzer UI.
    """
    st.header("🏦 Dormant Account Identification (CBUAE)")

    agent_options_dormant = [
        "📊 Comprehensive Dormancy Analysis (All Agents)",
        "--- Individual Agent Analysis ---",
        "🏦 Safe Deposit Box Dormancy (Art 2.6)",
        "📈 Investment Account Dormancy (Art 2.3)",
        "💰 Fixed Deposit Dormancy (Art 2.2)",
        "💳 Demand Deposit Dormancy (Art 2.1.1)",
        "📄 Unclaimed Payment Instruments (Art 2.4)",
        "🏛️ Central Bank Transfer Eligibility (Art 8.1-8.2)",
        "📞 Article 3 Process Requirements",
        "💎 High-Value Account Analysis",
        "🔄 Dormant-to-Active Transitions"
    ]

    selected_agent_dormant = st.selectbox(
        "Select Dormancy Analysis Task",
        agent_options_dormant,
        key="dormant_agent_selector_ui"
    )

    if selected_agent_dormant == "📊 Comprehensive Dormancy Analysis (All Agents)":
        render_summarized_dormant_analysis_view(df, report_date_str, llm, dormant_flags_history_df, config)
    elif selected_agent_dormant != "--- Individual Agent Analysis ---":
        render_individual_dormant_agent_view(df, selected_agent_dormant, report_date_str, llm, dormant_flags_history_df,
                                             config)


# --- Summarized View ---
def render_summarized_dormant_analysis_view(df, report_date_str, llm, dormant_flags_history_df, config):
    """
    Enhanced summarized view that includes comprehensive details from all dormant agents
    """
    st.subheader("📈 Comprehensive Dormancy Analysis Summary")

    if st.button("🚀 Run Complete Dormancy Analysis", key="run_summary_dormant_analysis_button"):
        with st.spinner("Running comprehensive dormancy analysis across all agents..."):
            results = run_all_dormant_identification_checks(
                df.copy(),
                report_date_str=report_date_str,
                llm_client=llm,
                config=config,
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
                help="Accounts with balance ≥ AED 100,000"
            )

        with kpi_cols[3]:
            st.metric(
                "CB Transfer Eligible",
                summary_kpis.get("count_eligible_for_cb_transfer", 0),
                help="Accounts eligible for Central Bank transfer"
            )

        # Detailed Agent Results
        st.subheader("🔍 Detailed Analysis by Regulatory Category")

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

                # Show details if available
                if agent_data['details'] and agent_data['count'] > 0:
                    st.markdown("**Processing Details:**")
                    detail_cols = st.columns(len(agent_data['details']))
                    for i, (key, value) in enumerate(agent_data['details'].items()):
                        if i < len(detail_cols):
                            with detail_cols[i]:
                                if isinstance(value, (int, float)):
                                    if 'balance' in key.lower() or 'value' in key.lower():
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
                st.caption(f"• Needs contact process: {art3_details.get('needs_initial_contact', 0)}")

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

        # Visual Charts
        st.subheader("📊 Visual Analytics")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Dormancy by CBUAE Article Category**")
            dormancy_data = {
                "Safe Deposit": summary_kpis.get("count_sdb_dormant", 0),
                "Investment": summary_kpis.get("count_investment_dormant", 0),
                "Fixed Deposit": summary_kpis.get("count_fixed_deposit_dormant", 0),
                "Demand Deposit": summary_kpis.get("count_demand_deposit_dormant", 0),
                "Unclaimed Instruments": summary_kpis.get("count_unclaimed_instruments", 0),
            }

            chart_df = pd.DataFrame(list(dormancy_data.items()), columns=['Category', 'Count'])
            chart_df = chart_df[chart_df['Count'] > 0]

            if not chart_df.empty:
                try:
                    import plotly.express as px
                    fig1 = px.pie(chart_df, values='Count', names='Category',
                                  title='Distribution of Dormant Items by CBUAE Category',
                                  color_discrete_sequence=px.colors.qualitative.Set3)
                    fig1.update_traces(textposition='inside', textinfo='percent+label')
                    fig1.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig1, use_container_width=True)
                except ImportError:
                    st.bar_chart(chart_df.set_index('Category'), height=300)
            else:
                st.info("No dormant accounts identified across categories.")

        with chart_col2:
            st.markdown("**Process Status Overview**")
            process_data = {
                "CB Transfer Eligible": summary_kpis.get("count_eligible_for_cb_transfer", 0),
                "Article 3 Process": summary_kpis.get("count_needing_art3_process", 0),
                "High Value (≥100k)": summary_kpis.get("count_high_value_dormant", 0),
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
                    fig2.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig2, use_container_width=True)
                except ImportError:
                    st.bar_chart(process_df.set_index('Status'), height=300)
            else:
                st.info("No process-specific items identified.")

        # AI-Generated Executive Summary
        st.subheader("🤖 AI-Generated Executive Summary & Strategic Insights")

        if llm:
            try:
                with st.spinner("Generating comprehensive AI analysis..."):
                    # Prepare comprehensive summary for AI analysis
                    comprehensive_summary = format_comprehensive_summary(results, summary_kpis, agent_summary)

                    # Use the prompt template
                    prompt_template = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
                    chain = prompt_template | llm | StrOutputParser()
                    ai_summary = chain.invoke({"analysis_details": comprehensive_summary})

                st.markdown(ai_summary)
                st.session_state.dormant_ai_summary_text_ui = ai_summary

                # Save to database
                try:
                    save_summary_to_db("dormant_analysis", ai_summary, results.get('report_date_used'))
                    st.success("Analysis saved to database!")
                except Exception as save_error:
                    st.warning(f"Could not save to database: {save_error}")

            except Exception as e:
                st.error(f"AI summary generation failed: {e}")
                fallback_summary = get_fallback_response("dormant_summary")
                comprehensive_summary = format_comprehensive_summary(results, summary_kpis, agent_summary)
                st.session_state.dormant_ai_summary_text_ui = f"{fallback_summary}\n\n{comprehensive_summary}"
                st.warning("Using fallback summary due to AI service unavailability.")
                st.text_area("Detailed Analysis", comprehensive_summary, height=300)
        else:
            st.error("LLM not available. Cannot generate AI analysis. Please ensure LLM is properly initialized.")
            comprehensive_summary = format_comprehensive_summary(results, summary_kpis, agent_summary)
            st.text_area("Raw Analysis Details", comprehensive_summary, height=400)
            st.session_state.dormant_ai_summary_text_ui = comprehensive_summary

        # Export Options
        st.subheader("📥 Export Comprehensive Report")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            # PDF Export
            comprehensive_summary = format_comprehensive_summary(results, summary_kpis, agent_summary)
            comprehensive_report_sections = [
                {"title": "Executive Summary",
                 "content": f"Total Accounts: {results.get('total_accounts_analyzed')}\nDormant Items: {summary_kpis.get('total_accounts_flagged_dormant', 0)}\nDormancy Rate: {summary_kpis.get('percentage_dormant_of_total', 0):.2f}%"},
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
            # CSV Export - create a summary DataFrame
            summary_data = []
            for agent_name, agent_data in agent_summary.items():
                summary_data.append({
                    'Category': agent_name,
                    'Count': agent_data['count'],
                    'Description': agent_data['desc']
                })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                download_csv_button(
                    summary_df,
                    "dormancy_analysis_summary.csv"
                )

        # Raw Data Explorer
        with st.expander("🔍 Raw Results Explorer (Technical Details)", expanded=False):
            result_category = st.selectbox(
                "Select Category to Explore:",
                ["summary_kpis", "sdb_dormant", "investment_dormant", "fixed_deposit_dormant",
                 "demand_deposit_dormant", "unclaimed_instruments", "eligible_for_cb_transfer",
                 "art3_process_needed", "high_value_dormant", "dormant_to_active"]
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
                f"🚨 **HIGH PRIORITY**: {summary_kpis.get('count_high_value_dormant', 0)} high-value dormant accounts identified (≥AED 100,000)")

        if summary_kpis.get("count_eligible_for_cb_transfer", 0) > 0:
            alerts.append(
                f"📋 **REGULATORY**: {summary_kpis.get('count_eligible_for_cb_transfer', 0)} accounts/instruments eligible for CBUAE transfer")

        if summary_kpis.get("count_needing_art3_process", 0) > 0:
            alerts.append(
                f"📞 **ACTION REQUIRED**: {summary_kpis.get('count_needing_art3_process', 0)} accounts need Article 3 process (customer contact)")

        if alerts:
            for alert in alerts:
                st.markdown(alert)
        else:
            st.success("✅ No immediate alerts. All dormancy processes appear to be under control.")

    else:
        st.info(
            "Click 'Run Complete Dormancy Analysis' to generate comprehensive dormancy insights across all CBUAE regulatory categories.")
        st.markdown("""
        **This comprehensive analysis will provide:**
        - 📊 Complete overview of all dormancy categories (CBUAE Arts 2.1-2.6, 8.1-8.2)
        - 🎯 Key performance indicators and regulatory metrics
        - 📈 Visual analytics with interactive charts
        - ⚡ Process status and required actions
        - 🤖 AI-generated strategic insights and recommendations
        - 📥 Exportable reports (PDF/CSV)
        - ⚠️ Priority alerts and regulatory compliance status
        """)


def format_comprehensive_summary(results, summary_kpis, agent_summary):
    """Format comprehensive summary for AI analysis."""
    total_balance = summary_kpis.get('total_dormant_balance_aed', 0)
    total_balance_formatted = f"AED {total_balance:,.0f}" if isinstance(total_balance, (int, float)) else str(
        total_balance)

    high_value_balance = summary_kpis.get('total_high_value_dormant_balance_aed', 0)
    high_value_balance_formatted = f"AED {high_value_balance:,.0f}" if isinstance(high_value_balance,
                                                                                  (int, float)) else str(
        high_value_balance)

    comprehensive_summary = f"""
CBUAE DORMANCY ANALYSIS COMPREHENSIVE REPORT
===========================================
Report Date: {results.get('report_date_used')}
Total Accounts Analyzed: {results.get('total_accounts_analyzed')}

EXECUTIVE SUMMARY:
- Total Dormant Items: {summary_kpis.get('total_accounts_flagged_dormant', 0)} ({summary_kpis.get('percentage_dormant_of_total', 0):.2f}%)
- Total Dormant Balance: {total_balance_formatted}
- High-Value Dormant: {summary_kpis.get('count_high_value_dormant', 0)} accounts ({high_value_balance_formatted})

DETAILED FINDINGS BY CBUAE REGULATORY CATEGORY:
"""

    for category, data in agent_summary.items():
        if not data['desc'].startswith("(Skipped"):
            comprehensive_summary += f"\n{category}:\n- {data['desc']}\n"
            if data['details']:
                for key, value in data['details'].items():
                    comprehensive_summary += f"  - {key.replace('_', ' ').title()}: {value}\n"

    comprehensive_summary += f"""
REGULATORY COMPLIANCE & ACTION ITEMS:
- Article 3 Process Required: {summary_kpis.get('count_needing_art3_process', 0)} accounts
- Central Bank Transfer Eligible: {summary_kpis.get('count_eligible_for_cb_transfer', 0)} items
- Dormant-to-Active Transitions: {summary_kpis.get('count_dormant_to_active_transitions', 0)} accounts

FINANCIAL IMPACT:
- Total Unclaimed Instruments Value: AED {summary_kpis.get('total_unclaimed_instruments_value', 0):,.0f}
- High-Value Dormant Balance: {high_value_balance_formatted}
"""
    return comprehensive_summary


# --- Individual Agent View ---
def render_individual_dormant_agent_view(df, selected_agent_key, report_date_str, llm, dormant_flags_history_df,
                                         config):
    st.subheader(f"🔍 Individual Agent Analysis: {selected_agent_key}")

    # Agent mapping
    agent_mapping = {
        "🏦 Safe Deposit Box Dormancy (Art 2.6)": ("safe_deposit_box_agent", AccountType.SAFE_DEPOSIT.value),
        "📈 Investment Account Dormancy (Art 2.3)": ("investment_account_agent", AccountType.INVESTMENT.value),
        "💰 Fixed Deposit Dormancy (Art 2.2)": ("fixed_deposit_agent", AccountType.FIXED_DEPOSIT.value),
        "💳 Demand Deposit Dormancy (Art 2.1.1)": ("demand_deposit_agent", AccountType.DEMAND_DEPOSIT.value),
        "📄 Unclaimed Payment Instruments (Art 2.4)": ("payment_instruments_agent",
                                                      AccountType.UNCLAIMED_INSTRUMENT.value),
        "🏛️ Central Bank Transfer Eligibility (Art 8.1-8.2)": ("cb_transfer_agent", None),
        "📞 Article 3 Process Requirements": ("article_3_process_agent", None),
        "💎 High-Value Account Analysis": ("high_value_account_agent", None),
        "🔄 Dormant-to-Active Transitions": ("transition_detection_agent", None)
    }

    agent_info = agent_mapping.get(selected_agent_key)
    if not agent_info:
        st.error(f"No agent mapping found for '{selected_agent_key}'")
        return

    agent_name, account_type_filter = agent_info

    if st.button(f"🚀 Run {selected_agent_key} Analysis", key=f"run_{agent_name}_button"):
        with st.spinner(f"Running {selected_agent_key} analysis..."):
            # Initialize the specific agent
            agents = initialize_dormant_agents(llm, config)
            agent = agents.get(agent_name)

            if agent is None:
                st.error(f"Failed to initialize agent: {agent_name}")
                return

            # Run analysis
            results, processed_count, description = run_single_agent_analysis(
                agent, df.copy(), account_type_filter
            )

            # Store results
            st.session_state[f"{agent_name}_results"] = {
                'results': results,
                'processed_count': processed_count,
                'description': description,
                'agent_name': agent_name,
                'selected_key': selected_agent_key
            }

        st.toast(f"{selected_agent_key} analysis complete!", icon="🔬")

    # Display results if available
    if f"{agent_name}_results" in st.session_state:
        stored_results = st.session_state[f"{agent_name}_results"]
        results = stored_results['results']
        processed_count = stored_results['processed_count']
        description = stored_results['description']

        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Items Identified", len(results))
        with col2:
            st.metric("Total Accounts Processed", processed_count)

        st.info(f"**Analysis Description:** {description}")

        # Results Summary
        if results:
            st.subheader("🎯 Analysis Results")

            # Create DataFrame for display
            display_data = []
            total_balance = 0

            for result in results[:10]:  # Show top 10 results
                account_data = result.get('account_data', {})
                balance = account_data.get('balance', 0)
                total_balance += balance

                display_data.append({
                    'Account ID': account_data.get('account_id', 'Unknown'),
                    'Status': result.get('status', 'Unknown'),
                    'Action': result.get('action', 'N/A'),
                    'Priority': result.get('priority', 'N/A'),
                    'Risk Level': result.get('risk_level', 'N/A'),
                    'Balance (AED)': f"{balance:,.0f}",
                    'Dormancy Days': result.get('dormancy_days', 0),
                    'Regulatory Citation': result.get('regulatory_citation', 'N/A')
                })

            if display_data:
                results_df = pd.DataFrame(display_data)
                st.dataframe(results_df, height=300, use_container_width=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Balance Affected", f"AED {total_balance:,.0f}")
                with col2:
                    avg_dormancy = sum(r.get('dormancy_days', 0) for r in results) / len(results) if results else 0
                    st.metric("Avg Dormancy Days", f"{avg_dormancy:.0f}")
                with col3:
                    high_priority_count = sum(1 for r in results if r.get('priority') in ['HIGH', 'CRITICAL'])
                    st.metric("High/Critical Priority", high_priority_count)

                # Export individual results
                download_csv_button(
                    results_df,
                    f"{agent_name}_analysis_results.csv"
                )

            # AI Insights for Individual Agent
            st.subheader(f"🤖 AI Insights for {selected_agent_key}")
            if llm and results:
                if st.button(f"Generate AI Insights", key=f"ai_btn_{agent_name}"):
                    with st.spinner("Generating AI insights..."):
                        try:
                            # Prepare data for AI analysis
                            sample_results = results[:5]  # Use first 5 results for AI analysis
                            analysis_context = f"""
Agent: {selected_agent_key}
Total Items Identified: {len(results)}
Total Balance Affected: AED {total_balance:,.0f}
Average Dormancy Days: {avg_dormancy:.0f}

Sample Results:
"""
                            for i, result in enumerate(sample_results, 1):
                                account_data = result.get('account_data', {})
                                analysis_context += f"""
Result {i}:
- Account: {account_data.get('account_id', 'Unknown')}
- Status: {result.get('status', 'Unknown')}
- Balance: AED {account_data.get('balance', 0):,.0f}
- Dormancy: {result.get('dormancy_days', 0)} days
- Priority: {result.get('priority', 'N/A')}
- Action: {result.get('action', 'N/A')}
"""

                            # Generate AI analysis using proper LangChain chains
                            obs_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                            obs_chain = obs_prompt | llm | StrOutputParser()
                            observations = obs_chain.invoke({"data": analysis_context})

                            trend_prompt = PromptTemplate.from_template(TREND_PROMPT)
                            trend_chain = trend_prompt | llm | StrOutputParser()
                            trends = trend_chain.invoke({"data": analysis_context})

                            narr_prompt = PromptTemplate.from_template(NARRATION_PROMPT)
                            narr_chain = narr_prompt | llm | StrOutputParser()
                            narration = narr_chain.invoke({"observation": observations, "trend": trends})

                            act_prompt = PromptTemplate.from_template(ACTION_PROMPT)
                            act_chain = act_prompt | llm | StrOutputParser()
                            actions = act_chain.invoke({"observation": observations, "trend": trends})

                            # Store AI results
                            st.session_state[f"ai_{agent_name}_obs"] = observations
                            st.session_state[f"ai_{agent_name}_trend"] = trends
                            st.session_state[f"ai_{agent_name}_narr"] = narration
                            st.session_state[f"ai_{agent_name}_act"] = actions

                        except Exception as e:
                            st.error(f"AI analysis failed: {e}")
                            # Use fallback response instead of mock
                            fallback_analysis = get_fallback_response("individual_agent_analysis")
                            st.session_state[
                                f"ai_{agent_name}_narr"] = f"{fallback_analysis}\n\nAnalysis Context:\n{analysis_context}"

                    st.toast("AI insights generated!", icon="💡")

                # Display AI insights if available
                if f"ai_{agent_name}_narr" in st.session_state:
                    ai_tabs = st.tabs(["📝 Narrative Summary", "🔍 Observations", "📈 Trends", "🚀 Actions"])

                    with ai_tabs[0]:
                        st.markdown(st.session_state[f"ai_{agent_name}_narr"])

                    with ai_tabs[1]:
                        st.markdown(st.session_state.get(f"ai_{agent_name}_obs", "No observations available"))

                    with ai_tabs[2]:
                        st.markdown(st.session_state.get(f"ai_{agent_name}_trend", "No trends available"))

                    with ai_tabs[3]:
                        st.markdown(st.session_state.get(f"ai_{agent_name}_act", "No actions available"))

                    # Export AI insights
                    ai_report_sections = [
                        {"title": f"Agent Analysis: {selected_agent_key}",
                         "content": f"Items identified: {len(results)}\nTotal balance: AED {total_balance:,.0f}"},
                        {"title": "AI Observations", "content": st.session_state.get(f"ai_{agent_name}_obs", "N/A")},
                        {"title": "AI Trends", "content": st.session_state.get(f"ai_{agent_name}_trend", "N/A")},
                        {"title": "AI Narrative", "content": st.session_state[f"ai_{agent_name}_narr"]},
                        {"title": "AI Actions", "content": st.session_state.get(f"ai_{agent_name}_act", "N/A")},
                    ]
                    download_pdf_button(
                        f"{selected_agent_key}_AI_Report",
                        ai_report_sections,
                        f"{agent_name}_ai_report.pdf"
                    )

            else:
                if not llm:
                    st.error("LLM not available for generating AI insights. Please ensure LLM is properly initialized.")
                else:
                    st.info("No results to analyze. Run the analysis first.")

            # Technical Details
            with st.expander("🔧 Technical Agent Details", expanded=False):
                if results:
                    st.json(results[0], expanded=True)  # Show first result structure
                else:
                    st.info("No technical details available - no results found.")

        else:
            st.info(f"No items identified by {selected_agent_key} analysis.")
            st.markdown("""
            **Possible reasons:**
            - No accounts meet the dormancy criteria for this category
            - All accounts in this category are active
            - Data may not contain the required fields for this analysis
            """)

    else:
        st.info(f"Click 'Run {selected_agent_key} Analysis' to execute the individual agent analysis.")

        # Show agent description
        agent_descriptions = {
            "🏦 Safe Deposit Box Dormancy (Art 2.6)": "Identifies Safe Deposit Boxes that have been inactive for 3+ years, requiring dormancy flagging per CBUAE Article 2.6.",
            "📈 Investment Account Dormancy (Art 2.3)": "Identifies Investment Accounts with no activity for 3+ years, subject to maturity-based dormancy rules per CBUAE Article 2.3.",
            "💰 Fixed Deposit Dormancy (Art 2.2)": "Identifies Fixed/Term Deposits that remain unclaimed 3+ years after maturity, requiring dormancy processing per CBUAE Article 2.2.",
            "💳 Demand Deposit Dormancy (Art 2.1.1)": "Identifies Demand Deposit accounts with no customer-initiated activity for 3+ years per CBUAE Article 2.1.1.",
            "📄 Unclaimed Payment Instruments (Art 2.4)": "Identifies unclaimed Bankers Cheques, Bank Drafts, and Cashier Orders outstanding for 1+ year per CBUAE Article 2.4.",
            "🏛️ Central Bank Transfer Eligibility (Art 8.1-8.2)": "Identifies accounts and balances eligible for transfer to CBUAE after 5+ years of dormancy per Articles 8.1-8.2.",
            "📞 Article 3 Process Requirements": "Identifies dormant accounts requiring or currently undergoing the Article 3 customer contact process.",
            "💎 High-Value Account Analysis": "Identifies high-value dormant accounts (≥AED 100,000) requiring special attention and escalated processing.",
            "🔄 Dormant-to-Active Transitions": "Identifies accounts that were previously flagged as dormant but have shown recent activity, requiring status updates."
        }

        description = agent_descriptions.get(selected_agent_key, "No description available.")
        st.info(f"**Agent Description:** {description}")