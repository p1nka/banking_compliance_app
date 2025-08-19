# --- START OF FILE dormant_ui.py ---

import streamlit as st
from datetime import datetime, timedelta, date  # Ensure 'date' is imported
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
# Ensure these modules and their contents (like DORMANT_SUMMARY_PROMPT) are present.
# If you don't have a database or specific LLM prompt files, you might need to mock them.
try:
    from database.pipelines import AgentDatabasePipeline
except ImportError:
    class AgentDatabasePipeline:
        def __init__(self, *args, **kwargs): pass

        def save_analysis(self, *args, **kwargs): logging.warning("AgentDatabasePipeline not available.")


from data.exporters import download_pdf_button, download_csv_button


try:
    from ai.llm import (
        DORMANT_SUMMARY_PROMPT,
        OBSERVATION_PROMPT,
        TREND_PROMPT,
        NARRATION_PROMPT,
        ACTION_PROMPT
    )
except ImportError:
    # Define placeholder prompts if the file doesn't exist
    DORMANT_SUMMARY_PROMPT = """Analyze the following banking dormancy report: {analysis_details}\n\nProvide an executive summary, key observations, potential trends, and actionable recommendations focusing on CBUAE compliance and risk mitigation."""
    OBSERVATION_PROMPT = """Based on the following data, what are the key observations?\nData: {data}"""
    TREND_PROMPT = """Given these observations, what potential trends can be identified?\nObservation: {observation}"""
    NARRATION_PROMPT = """Combine these observations and trends into a concise narrative summary.\nObservation: {observation}\nTrend: {trend}"""
    ACTION_PROMPT = """Based on the observations and trends, what immediate and long-term actions are recommended?\nObservation: {observation}\nTrend: {trend}"""
    logging.warning("AI LLM prompt templates not found, using default placeholders.")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from database.operations import save_summary_to_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default values for compliance thresholds (if needed outside render_compliance_thresholds)
DEFAULT_DORMANT_DAYS = 1095  # 3 years for general dormancy
DEFAULT_FREEZE_DAYS = 1825  # 5 years for freeze/transfer
DEFAULT_CBUAE_DATE = "2020-01-01"  # Arbitrary past date


def download_csv_button(df, file_name):
    @st.cache_data
    def convert_df(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv = convert_df(df)
    st.download_button(
        label=f"Download Results as CSV",
        data=csv,
        file_name=file_name,
        mime='text/csv',
    )


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


# COMPLETE FIX for ui/dormant_ui.py based on your actual CSV data
# Replace the existing functions with these corrected versions

def prepare_account_data(df_row, account_type=None):
    """Prepare account data - FIXED for your actual CSV structure."""

    # Your CSV uses exact column names, so we can access them directly
    account_data = {
        # Core identifiers - direct mapping to your CSV columns
        'account_id': str(df_row.get('account_id', 'Unknown')),
        'account_type': str(df_row.get('account_type', 'unknown')).upper(),
        'account_subtype': str(df_row.get('account_subtype', '')).upper(),

        # Financial data
        'balance': float(df_row.get('balance_current', 0) or 0),
        'balance_current': float(df_row.get('balance_current', 0) or 0),

        # Key date fields for dormancy analysis
        'last_activity_date': df_row.get('last_transaction_date'),
        'last_transaction_date': df_row.get('last_transaction_date'),
        'maturity_date': df_row.get('maturity_date'),
        'opening_date': df_row.get('opening_date'),

        # Dormancy specific fields
        'dormancy_status': str(df_row.get('dormancy_status', 'active')).upper(),
        'dormancy_trigger_date': df_row.get('dormancy_trigger_date'),
        'transfer_eligibility_date': df_row.get('transfer_eligibility_date'),
        'current_stage': str(df_row.get('current_stage', '')).upper(),

        # Contact and process fields
        'contact_attempts_made': int(df_row.get('contact_attempts_made', 0) or 0),
        'last_contact_date': df_row.get('last_contact_date'),
        'last_contact_attempt_date': df_row.get('last_contact_attempt_date'),

        # Customer information
        'customer_type': str(df_row.get('customer_type', 'INDIVIDUAL')).upper(),
        'customer_tier': str(df_row.get('customer_type', 'standard')).lower(),

        # Additional fields that agents might need
        'currency': str(df_row.get('currency', 'AED')),
        'account_status': str(df_row.get('account_status', 'ACTIVE')).upper(),
        'auto_renewal': str(df_row.get('auto_renewal', '')).upper(),

        # For transition detection
        'previous_dormancy_status': str(df_row.get('dormancy_status', 'active')).lower(),
        'current_activity_status': str(df_row.get('account_status', 'active')).lower(),
    }

    # Convert date fields properly
    date_fields = ['last_activity_date', 'last_transaction_date', 'maturity_date',
                   'opening_date', 'dormancy_trigger_date', 'last_contact_date',
                   'last_contact_attempt_date', 'transfer_eligibility_date']

    for date_field in date_fields:
        if account_data[date_field]:
            account_data[date_field] = convert_date_safely(account_data[date_field])

    return account_data


def convert_date_safely(date_value):
    """Convert date field with support for your CSV date format."""
    if pd.isna(date_value) or date_value is None or str(date_value).strip() == '':
        return None

    if isinstance(date_value, datetime):
        return date_value

    if isinstance(date_value, str):
        date_value = date_value.strip()
        if not date_value:
            return None

        # Your CSV uses YYYY-MM-DD format, try this first
        try:
            return datetime.strptime(date_value, "%Y-%m-%d")
        except ValueError:
            pass

        # Try other common formats
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_value, fmt)
            except ValueError:
                continue

        # Try pandas parser as last resort
        try:
            return pd.to_datetime(date_value)
        except:
            return None

    return None


def run_single_agent_analysis(agent, df, report_date: datetime, account_type_filter=None):
    """Run agent analysis - FINAL FIX with corrected filtering logic."""
    results = []
    processed_count = 0

    if agent is None:
        return results, processed_count, "Agent not initialized"

    agent_name = agent.__class__.__name__

    st.write(f"🔍 Running {agent_name} on {len(df)} accounts")

    # Debug: Show what's in the data
    if not df.empty:
        unique_types = df['account_type'].unique() if 'account_type' in df.columns else ['N/A']
        unique_subtypes = df['account_subtype'].unique() if 'account_subtype' in df.columns else ['N/A']
        st.write(f"📊 Account types: {', '.join(unique_types)}")
        st.write(f"📊 Account subtypes: {', '.join(unique_subtypes)}")

    for idx, row in df.iterrows():
        try:
            account_data = prepare_account_data(row)

            # UPDATED FILTERING LOGIC - More inclusive based on your actual data
            should_process = True  # Default to process all

            if agent_name == "SafeDepositBoxAgent":
                # Look for SDB_LINKED subtype (you have this in your data)
                subtype = account_data.get('account_subtype', '').upper()
                should_process = (subtype == 'SDB_LINKED')

            elif agent_name == "PaymentInstrumentsAgent":
                # Look for INSTRUMENT_LINKED subtype (you have this in your data)
                subtype = account_data.get('account_subtype', '').upper()
                should_process = (subtype == 'INSTRUMENT_LINKED')

            elif agent_name == "InvestmentAccountAgent":
                # Process INVESTMENT account types (you have this in your data)
                acct_type = account_data.get('account_type', '').upper()
                should_process = (acct_type == 'INVESTMENT')

            elif agent_name == "FixedDepositAgent":
                # Process FIXED_DEPOSIT account types (you have this in your data)
                acct_type = account_data.get('account_type', '').upper()
                should_process = (acct_type == 'FIXED_DEPOSIT')

            elif agent_name == "DemandDepositAgent":
                # Process CURRENT and SAVINGS (most of your accounts)
                # Exclude special subtypes to avoid double counting
                acct_type = account_data.get('account_type', '').upper()
                subtype = account_data.get('account_subtype', '').upper()
                should_process = (acct_type in ['CURRENT', 'SAVINGS'] and
                                  subtype not in ['SDB_LINKED', 'INSTRUMENT_LINKED'])

            elif agent_name == "CBTransferAgent":
                # Process ALL dormant accounts for transfer eligibility
                should_process = True

            elif agent_name == "Article3ProcessAgent":
                # Process ALL dormant accounts for contact requirements
                should_process = True

            elif agent_name == "HighValueAccountAgent":
                # Process ALL accounts to check for high value + dormant
                should_process = True

            elif agent_name == "TransitionDetectionAgent":
                # Process ALL accounts to detect transitions
                should_process = True



            if not should_process:
                processed_count += 1
                continue

            # Execute the agent logic
            result = agent.execute(account_data, report_date)



            # Check for meaningful results
            if result and 'error' not in result:
                is_meaningful = False

                # Check for meaningful results based on agent type
                if 'status' in result:
                    if result['status'] in [
                        ActivityStatus.DORMANT.value,
                        ActivityStatus.UNCLAIMED.value,
                        ActivityStatus.PENDING_REVIEW.value,
                        'Process Pending',
                        'High Value Dormant',
                        'Reactivated'
                    ]:
                        is_meaningful = True

                # For CB Transfer Agent
                if 'eligible' in result and result['eligible']:
                    is_meaningful = True

                # For Article 3 Process Agent
                if agent_name == 'Article3ProcessAgent' and result.get('status') == 'Process Pending':
                    is_meaningful = True

                if is_meaningful:
                    result['account_data'] = account_data
                    results.append(result)

            processed_count += 1

        except Exception as e:
            st.error(f"Error processing account {row.get('account_id', 'Unknown')} with agent {agent_name}: {e}")
            import traceback
            st.write(f"Full error: {traceback.format_exc()}")

    st.write(f"✅ {agent_name}: Processed {processed_count} accounts, found {len(results)} meaningful results")

    # Additional debug for zero results
    if len(results) == 0 and processed_count > 0:
        st.warning(f"⚠️ {agent_name} processed {processed_count} accounts but found 0 results. This might indicate:")
        st.write("1. Accounts don't meet the dormancy criteria for this agent")
        st.write("2. Date calculations aren't working properly")
        st.write("3. Agent logic needs adjustment")

        # Show sample account data for debugging
        if not df.empty:
            sample_account = prepare_account_data(df.iloc[0])
            st.write(f"Sample account data: {sample_account}")

    return results, processed_count, f"Processed {processed_count} accounts"


def run_all_dormant_identification_checks(df, report_date_str, llm_client, config, dormant_flags_history_df=None):
    """Run comprehensive dormancy analysis using all agents."""

    # Create default empty results structure for safe return
    default_results = {
        'report_date_used': report_date_str,
        'total_accounts_analyzed': len(df) if df is not None else 0,
        'summary_kpis': {
            'total_accounts_flagged_dormant': 0, 'percentage_dormant_of_total': 0.0, 'total_dormant_balance_aed': 0.0,
            'count_high_value_dormant': 0, 'total_high_value_dormant_balance_aed': 0.0,
            'count_eligible_for_cb_transfer': 0, 'count_sdb_dormant': 0, 'count_investment_dormant': 0,
            'count_fixed_deposit_dormant': 0, 'count_demand_deposit_dormant': 0, 'count_unclaimed_instruments': 0,
            'count_needing_art3_process': 0, 'count_needing_proactive_contact': 0,
            'count_dormant_to_active_transitions': 0, 'total_unclaimed_instruments_value': 0.0
        },
        'sdb_dormant': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'investment_dormant': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'fixed_deposit_dormant': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'demand_deposit_dormant': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'unclaimed_instruments': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'eligible_for_cb_transfer': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'art3_process_needed': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'high_value_dormant': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'dormant_to_active': {'count': 0, 'desc': 'Analysis not performed', 'details': {}},
        'proactive_contact_needed': {'count': 0, 'desc': 'Analysis not performed', 'details': {}}
    }

    if llm_client is None or config is None:
        st.error("LLM client and config are required for dormancy analysis. Please ensure proper initialization.")
        default_results['error'] = "Missing LLM client or config"
        return default_results

    if df is None or df.empty:
        st.error("No data available for analysis.")
        default_results['error'] = "No data available"
        return default_results

    try:
        agents = initialize_dormant_agents(llm_client, config)
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        default_results['error'] = f"Agent initialization failed: {e}"
        return default_results

    report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
    total_accounts = len(df)

    results = {
        'report_date_used': report_date_str,
        'total_accounts_analyzed': total_accounts,
        'summary_kpis': {}  # This will be populated later
    }

    # Initialize lists to collect results from each agent
    sdb_results, inv_results, fd_results, dd_results, pi_results, cb_results, art3_results, hv_results, td_results = [], [], [], [], [], [], [], [], []

    # Safe Deposit Box Analysis
    try:
        sdb_results, sdb_count, sdb_desc = run_single_agent_analysis(agents.get('safe_deposit_box_agent'), df,
                                                                     report_date, AccountType.SAFE_DEPOSIT.value)
        results['sdb_dormant'] = {'count': len(sdb_results),
                                  'desc': f"Safe Deposit Boxes flagged as dormant (3+ years inactive): {len(sdb_results)}",
                                  'details': {'total_processed': sdb_count}}
        results['summary_kpis']['count_sdb_dormant'] = len(sdb_results)
    except Exception as e:
        logger.error(f"Safe Deposit Box analysis failed: {e}")
        results['sdb_dormant'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    # Investment Account Analysis
    try:
        inv_results, inv_count, inv_desc = run_single_agent_analysis(agents.get('investment_account_agent'), df,
                                                                     report_date, AccountType.INVESTMENT.value)
        results['investment_dormant'] = {'count': len(inv_results),
                                         'desc': f"Investment accounts flagged as dormant (3+ years inactive): {len(inv_results)}",
                                         'details': {'total_processed': inv_count}}
        results['summary_kpis']['count_investment_dormant'] = len(inv_results)
    except Exception as e:
        logger.error(f"Investment account analysis failed: {e}")
        results['investment_dormant'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    # Fixed Deposit Analysis
    try:
        fd_results, fd_count, fd_desc = run_single_agent_analysis(agents.get('fixed_deposit_agent'), df, report_date,
                                                                  AccountType.FIXED_DEPOSIT.value)
        results['fixed_deposit_dormant'] = {'count': len(fd_results),
                                            'desc': f"Fixed deposits unclaimed post-maturity (3+ years): {len(fd_results)}",
                                            'details': {'total_processed': fd_count}}
        results['summary_kpis']['count_fixed_deposit_dormant'] = len(fd_results)
    except Exception as e:
        logger.error(f"Fixed deposit analysis failed: {e}")
        results['fixed_deposit_dormant'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    # Demand Deposit Analysis
    try:
        dd_results, dd_count, dd_desc = run_single_agent_analysis(agents.get('demand_deposit_agent'), df, report_date,
                                                                  AccountType.DEMAND_DEPOSIT.value)
        results['demand_deposit_dormant'] = {'count': len(dd_results),
                                             'desc': f"Demand deposits flagged as dormant (3+ years inactive): {len(dd_results)}",
                                             'details': {'total_processed': dd_count}}
        results['summary_kpis']['count_demand_deposit_dormant'] = len(dd_results)
    except Exception as e:
        logger.error(f"Demand deposit analysis failed: {e}")
        results['demand_deposit_dormant'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    # Payment Instruments Analysis
    try:
        pi_results, pi_count, pi_desc = run_single_agent_analysis(agents.get('payment_instruments_agent'), df,
                                                                  report_date, AccountType.UNCLAIMED_INSTRUMENT.value)
        results['unclaimed_instruments'] = {'count': len(pi_results),
                                            'desc': f"Unclaimed payment instruments (1+ year): {len(pi_results)}",
                                            'details': {'total_processed': pi_count}}
        results['summary_kpis']['count_unclaimed_instruments'] = len(pi_results)
    except Exception as e:
        logger.error(f"Payment instruments analysis failed: {e}")
        results['unclaimed_instruments'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    # Central Bank Transfer Eligibility
    try:
        cb_results, cb_count, cb_desc = run_single_agent_analysis(agents.get('cb_transfer_agent'), df, report_date)
        eligible_cb_results = [r for r in cb_results if r.get('eligible', False)]
        results['eligible_for_cb_transfer'] = {'count': len(eligible_cb_results),
                                               'desc': f"Items eligible for CBUAE transfer (5+ years dormant): {len(eligible_cb_results)}",
                                               'details': {'total_processed': cb_count}}
        results['summary_kpis']['count_eligible_for_cb_transfer'] = len(eligible_cb_results)
    except Exception as e:
        logger.error(f"CB transfer analysis failed: {e}")
        results['eligible_for_cb_transfer'] = {'count': 0, 'desc': f"Analysis failed: {e}",
                                               'details': {'error': str(e)}}
        eligible_cb_results = []  # Ensure this list is defined for KPI aggregation

    # Article 3 Process Analysis
    try:
        art3_results, art3_count, art3_desc = run_single_agent_analysis(agents.get('article_3_process_agent'), df,
                                                                        report_date)
        needing_art3 = [r for r in art3_results if r.get('status') == 'Process Pending']
        results['art3_process_needed'] = {'count': len(needing_art3),
                                          'desc': f"Accounts requiring Article 3 process (contact procedures): {len(needing_art3)}",
                                          'details': {'total_processed': art3_count,
                                                      'needs_initial_contact': len(needing_art3)}}
        results['summary_kpis']['count_needing_art3_process'] = len(needing_art3)
    except Exception as e:
        logger.error(f"Article 3 process analysis failed: {e}")
        results['art3_process_needed'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    # High Value Account Analysis
    try:
        hv_results, hv_count, hv_desc = run_single_agent_analysis(agents.get('high_value_account_agent'), df,
                                                                  report_date)
        high_value_dormant = [r for r in hv_results if r.get('status') == 'High Value Dormant']
        total_hv_balance = sum(r.get('account_data', {}).get('balance', 0) for r in high_value_dormant)

        results['high_value_dormant'] = {'count': len(high_value_dormant),
                                         'desc': f"High-value dormant accounts (≥AED 100,000): {len(high_value_dormant)}",
                                         'details': {'total_processed': hv_count, 'total_balance': total_hv_balance}}
        results['summary_kpis']['count_high_value_dormant'] = len(high_value_dormant)
        results['summary_kpis']['total_high_value_dormant_balance_aed'] = total_hv_balance
    except Exception as e:
        logger.error(f"High value account analysis failed: {e}")
        results['high_value_dormant'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}
        high_value_dormant = []  # Ensure this list is defined for KPI aggregation

    # Transition Detection Analysis
    try:
        td_results, td_count, td_desc = run_single_agent_analysis(agents.get('transition_detection_agent'), df,
                                                                  report_date)
        reactivated = [r for r in td_results if r.get('status') == 'Reactivated']
        results['dormant_to_active'] = {'count': len(reactivated),
                                        'desc': f"Accounts reactivated (dormant to active): {len(reactivated)}",
                                        'details': {'total_processed': td_count}}
        results['summary_kpis']['count_dormant_to_active_transitions'] = len(reactivated)
    except Exception as e:
        logger.error(f"Transition detection analysis failed: {e}")
        results['dormant_to_active'] = {'count': 0, 'desc': f"Analysis failed: {e}", 'details': {'error': str(e)}}

    results['proactive_contact_needed'] = {'count': 0,
                                           'desc': "Accounts nearing dormancy requiring proactive contact: 0",
                                           'details': {'total_processed': 0}}
    results['summary_kpis']['count_needing_proactive_contact'] = 0

    # Calculate overall KPIs
    try:
        # Aggregate all relevant results for total dormant calculation
        # Make sure all lists are defined even if previous try-except blocks failed
        all_flagged_results = (
                sdb_results + inv_results + fd_results + dd_results + pi_results +
                high_value_dormant + eligible_cb_results + needing_art3
        # Art3 flagged are also "dormant" from process perspective
        )

        unique_account_ids = set()
        total_dormant_balance = 0.0

        # Keep track of accounts whose balance has already been added to avoid double counting
        processed_balance_account_ids = set()

        for r in all_flagged_results:
            acc_id = r.get('account_data', {}).get('account_id')
            if acc_id:
                unique_account_ids.add(acc_id)
                if acc_id not in processed_balance_account_ids:
                    total_dormant_balance += r.get('account_data', {}).get('balance', 0.0)
                    processed_balance_account_ids.add(acc_id)

        results['summary_kpis']['total_accounts_flagged_dormant'] = len(unique_account_ids)
        results['summary_kpis']['percentage_dormant_of_total'] = (
                    len(unique_account_ids) / total_accounts * 100) if total_accounts > 0 else 0.0
        results['summary_kpis']['total_dormant_balance_aed'] = total_dormant_balance

        results['summary_kpis']['total_unclaimed_instruments_value'] = sum(
            r.get('account_data', {}).get('balance', 0.0) for r in pi_results)
    except Exception as e:
        logger.error(f"Error calculating overall KPIs: {e}", exc_info=True)
        # Re-initialize KPIs to 0 if calculation failed to prevent partial/incorrect display
        results['summary_kpis'] = default_results['summary_kpis']

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

        if results is None:
            st.error("Analysis failed to complete. Please check your configuration and try again.")
            return

        if 'error' in results:
            st.error(f"Analysis failed: {results['error']}")
            st.info("Please check that your LLM and configuration are properly initialized.")
            return

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

        # Create agent summary data (THIS IS THE SECTION WHERE THE 'icon' KEY WAS MISSING BEFORE)
        agent_summary = {
            "Safe Deposit Boxes (Art 2.6)": {
                "count": results["sdb_dormant"]["count"],
                "desc": results["sdb_dormant"]["desc"],
                "details": results["sdb_dormant"]["details"],
                "icon": "🏦",  # Added icon
                "color": "red" if results["sdb_dormant"]["count"] > 0 else "gray"  # Added color
            },
            "Investment Accounts (Art 2.3)": {
                "count": results["investment_dormant"]["count"],
                "desc": results["investment_dormant"]["desc"],
                "details": results["investment_dormant"]["details"],
                "icon": "📈",  # Added icon
                "color": "orange" if results["investment_dormant"]["count"] > 0 else "gray"  # Added color
            },
            "Fixed Deposits (Art 2.2)": {
                "count": results["fixed_deposit_dormant"]["count"],
                "desc": results["fixed_deposit_dormant"]["desc"],
                "details": results["fixed_deposit_dormant"]["details"],
                "icon": "💰",  # Added icon
                "color": "yellow" if results["fixed_deposit_dormant"]["count"] > 0 else "gray"  # Added color
            },
            "Demand Deposits (Art 2.1.1)": {
                "count": results["demand_deposit_dormant"]["count"],
                "desc": results["demand_deposit_dormant"]["desc"],
                "details": results["demand_deposit_dormant"]["details"],
                "icon": "💳",  # Added icon
                "color": "blue" if results["demand_deposit_dormant"]["count"] > 0 else "gray"  # Added color
            },
            "Unclaimed Payment Instruments (Art 2.4)": {
                "count": results["unclaimed_instruments"]["count"],
                "desc": results["unclaimed_instruments"]["desc"],
                "details": results["unclaimed_instruments"]["details"],
                "icon": "📄",  # Added icon
                "color": "purple" if results["unclaimed_instruments"]["count"] > 0 else "gray"  # Added color
            }
        }

        # Display agent results in expandable sections
        for agent_name, agent_data in agent_summary.items():
            with st.expander(f"{agent_data['icon']} {agent_name} - {agent_data['count']} items",
                             expanded=agent_data['count'] > 0):

                col_desc, col_metric = st.columns([3, 1])
                with col_desc:
                    if not agent_data['desc'].startswith("(Skipped") and not agent_data['desc'].startswith(
                            "Analysis failed"):
                        st.write(agent_data['desc'])
                    else:
                        st.warning(agent_data['desc'])

                with col_metric:
                    st.metric("Count", agent_data['count'])

                # Show details if available
                if agent_data['details'] and agent_data['count'] > 0 and 'error' not in agent_data['details']:
                    st.markdown("**Processing Details:**")
                    # Display details more cleanly, perhaps as a list or small table
                    for key, value in agent_data['details'].items():
                        st.write(f"- **{key.replace('_', ' ').title()}**: {value}")

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
                art3_details = results.get("art3_process_needed", {}).get("details", {})
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
                st.info("No dormant accounts identified across categories for charting.")

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
                st.info("No process-specific items identified for charting.")

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
                    save_summary_to_db(
                        observation="dormant_analysis",
                        trend=ai_summary,
                        insight=comprehensive_summary,
                        action="Review dormant accounts and implement recommended actions"
                    )
                    st.success("Analysis saved to database!")
                except Exception as save_error:
                    st.warning(f"Could not save to database: {save_error}")

            except Exception as e:
                st.error(f"AI summary generation failed: {e}")
                comprehensive_summary = format_comprehensive_summary(results, summary_kpis, agent_summary)
                st.session_state.dormant_ai_summary_text_ui = comprehensive_summary
                st.warning("AI service unavailable or failed. Showing detailed analysis without AI insights.")
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
            ai_summary_content = st.session_state.get("dormant_ai_summary_text_ui", "AI Summary not available.")

            # Combine all parts for PDF
            pdf_content = f"""
            # Comprehensive Dormancy Analysis Report - {results.get('report_date_used')}

            ## Executive Summary
            Total Accounts Analyzed: {results.get('total_accounts_analyzed')}
            Total Flagged Dormant: {summary_kpis.get('total_accounts_flagged_dormant', 0)} ({summary_kpis.get('percentage_dormant_of_total', 0):.2f}%)
            Total Dormant Balance: AED {summary_kpis.get('total_dormant_balance_aed', 0):,.0f}

            ## AI Strategic Insights
            {ai_summary_content}

            ## Detailed Analysis by Category
            {comprehensive_summary}
            """

            download_pdf_button(
                "Comprehensive Dormancy Analysis Report",
                [{"title": "Comprehensive Report", "content": pdf_content}],
                "comprehensive_dormancy_report.pdf"
            )

        with export_col2:
            # CSV Export - create a summary DataFrame
            summary_data = []
            for agent_name, agent_data in agent_summary.items():
                summary_data.append({
                    'Category': agent_name,
                    'Count': agent_data['count'],
                    'Description': agent_data['desc'],
                    'Total_Processed': agent_data['details'].get('total_processed', 'N/A'),
                    'Error': agent_data['details'].get('error', 'N/A')
                })

            # Add other KPIs to the summary dataframe if they are scalar
            summary_data.append({'Category': 'Overall Total Flagged Dormant',
                                 'Count': summary_kpis.get('total_accounts_flagged_dormant', 0), 'Description': '',
                                 'Total_Processed': '', 'Error': ''})
            summary_data.append(
                {'Category': 'Total Dormant Balance (AED)', 'Count': summary_kpis.get('total_dormant_balance_aed', 0),
                 'Description': '', 'Total_Processed': '', 'Error': ''})
            summary_data.append(
                {'Category': 'High-Value Dormant Count', 'Count': summary_kpis.get('count_high_value_dormant', 0),
                 'Description': '', 'Total_Processed': '', 'Error': ''})
            summary_data.append({'Category': 'CB Transfer Eligible Count',
                                 'Count': summary_kpis.get('count_eligible_for_cb_transfer', 0), 'Description': '',
                                 'Total_Processed': '', 'Error': ''})

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
        if not data['desc'].startswith("(Skipped") and not data['desc'].startswith("Analysis failed"):
            comprehensive_summary += f"\n{category}:\n- {data['desc']}\n"
            if data['details'] and 'error' not in data['details']:
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

            report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
            # Run analysis
            results, processed_count, description = run_single_agent_analysis(
                agent, df.copy(), report_date, account_type_filter
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

            # Only show relevant columns based on agent type and data availability
            for result in results:
                account_data = result.get('account_data', {})
                balance = account_data.get('balance', 0)
                total_balance += balance

                row_data = {
                    'Account ID': account_data.get('account_id', 'Unknown'),
                    'Status': result.get('status', 'Unknown'),
                    'Action': result.get('action', 'N/A'),
                    'Priority': result.get('priority', 'N/A'),
                    'Risk Level': result.get('risk_level', 'N/A'),
                    'Balance (AED)': f"{balance:,.0f}",
                }

                # Add agent-specific fields if available and relevant
                if 'dormancy_days' in result:
                    row_data['Dormancy Days'] = result['dormancy_days']
                if 'maturity_date' in account_data and account_data['maturity_date'] is not None:
                    row_data['Maturity Date'] = account_data['maturity_date'].strftime('%Y-%m-%d')
                if 'last_activity_date' in account_data and account_data['last_activity_date'] is not None:
                    row_data['Last Activity'] = account_data['last_activity_date'].strftime('%Y-%m-%d')
                if 'regulatory_citation' in result:
                    row_data['Regulatory Citation'] = result['regulatory_citation']
                if 'contact_attempts_made' in account_data:
                    row_data['Contact Attempts'] = account_data['contact_attempts_made']
                if 'eligible' in result:  # For CBTransferAgent
                    row_data['Eligible for CB Transfer'] = 'Yes' if result['eligible'] else 'No'

                display_data.append(row_data)

            if display_data:
                results_df = pd.DataFrame(display_data)
                st.dataframe(results_df, height=300, use_container_width=True)

                # REPLACE the summary metrics section (around line 800+) with this:

                # Summary metrics - FIXED VERSION
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Balance Affected", f"AED {total_balance:,.0f}")

                with col2:
                    # SAFE dormancy calculation with multiple fallback methods
                    avg_dormancy = 0
                    dormancy_count = 0

                    try:
                        # Method 1: Get dormancy from agent results (most reliable)
                        valid_dormancy_values = []
                        for result in results:
                            dormancy_days = result.get('dormancy_days')
                            if dormancy_days is not None and isinstance(dormancy_days,
                                                                        (int, float)) and dormancy_days > 0:
                                valid_dormancy_values.append(dormancy_days)

                        if valid_dormancy_values:
                            avg_dormancy = sum(valid_dormancy_values) / len(valid_dormancy_values)
                            dormancy_count = len(valid_dormancy_values)

                        # Method 2: Fallback to display data if agent results don't have dormancy_days
                        elif display_data and len(display_data) > 0:
                            # Check if Dormancy Days column exists in display data
                            if 'Dormancy Days' in display_data[0]:
                                df_temp = pd.DataFrame(display_data)
                                dormancy_series = pd.to_numeric(df_temp['Dormancy Days'], errors='coerce')
                                valid_dormancy_series = dormancy_series.dropna()

                                if len(valid_dormancy_series) > 0:
                                    avg_dormancy = valid_dormancy_series.mean()
                                    dormancy_count = len(valid_dormancy_series)

                        # Display the metric
                        if avg_dormancy > 0 and dormancy_count > 0:
                            st.metric(
                                "Avg Dormancy Days",
                                f"{avg_dormancy:.0f}",
                                help=f"Based on {dormancy_count} accounts with dormancy data"
                            )
                        else:
                            st.metric(
                                "Avg Dormancy Days",
                                "N/A",
                                help="No valid dormancy data found in results"
                            )

                    except Exception as e:
                        st.metric(
                            "Avg Dormancy Days",
                            "Error",
                            help=f"Calculation error: {str(e)}"
                        )

                with col3:
                    high_priority_count = sum(
                        1 for r in results
                        if r.get('priority', '').upper() in ['HIGH', 'CRITICAL', 'IMMEDIATE','MEDIUM']
                    )
                    st.metric("High/Critical Priority", high_priority_count)

            # AI Insights for Individual Agent
            st.subheader(f"🤖 AI Insights for {selected_agent_key}")
            if llm and results:
                if st.button(f"Generate AI Insights", key=f"ai_btn_{agent_name}"):
                    with st.spinner("Generating AI insights..."):
                        try:
                            # Prepare data for AI analysis
                            # Use max 5 results or less for AI context to avoid hitting token limits
                            sample_results_for_ai = results[:5]
                            analysis_context_parts = []
                            analysis_context_parts.append(f"Agent: {selected_agent_key}")
                            analysis_context_parts.append(f"Total Items Identified: {len(results)}")
                            analysis_context_parts.append(f"Total Balance Affected: AED {total_balance:,.0f}")
                            if 'dormancy_days' in display_data[0]:
                                analysis_context_parts.append(f"Average Dormancy Days: {avg_dormancy:.0f}")

                            analysis_context_parts.append("\nSample Results:")
                            for i, result in enumerate(sample_results_for_ai, 1):
                                account_data = result.get('account_data', {})
                                analysis_context_parts.append(f"""
Result {i}:
- Account: {account_data.get('account_id', 'Unknown')}
- Status: {result.get('status', 'Unknown')}
- Balance: AED {account_data.get('balance', 0):,.0f}
- Dormancy: {result.get('dormancy_days', 0) if 'dormancy_days' in result else 'N/A'} days
- Priority: {result.get('priority', 'N/A')}
- Action: {result.get('action', 'N/A')}
""")
                            analysis_context = "\n".join(analysis_context_parts)

                            # Generate AI analysis using proper LangChain chains
                            obs_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                            obs_chain = obs_prompt | llm | StrOutputParser()
                            observations = obs_chain.invoke({"data": analysis_context})

                            trend_prompt = PromptTemplate.from_template(TREND_PROMPT)
                            trend_chain = trend_prompt | llm | StrOutputParser()
                            trends = trend_chain.invoke(
                                {"data": analysis_context})  # Pass context if trend doesn't use observations

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
                            st.session_state[
                                f"ai_{agent_name}_narr"] = f"AI analysis unavailable. Raw analysis:\n\n{analysis_context}"
                            logger.error(f"AI generation for {agent_name} failed: {e}", exc_info=True)

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
