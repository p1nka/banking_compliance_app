
"""
Connector Module for Dormant and Compliance Agents

This module provides functions that connect the outputs of dormant agents
with the inputs required by compliance agents. It handles data transformation,
aggregation, and flag propagation between agent systems.
"""

import pandas as pd
from datetime import datetime, timedelta
from compliance import ComplianceOrchestrator
from dormant import DormantAccountOrchestrator


def process_dormant_results_for_compliance(dormant_results, df):
    """
    Process the results of dormant identification agents to prepare the data
    for compliance agents.

    Args:
        dormant_results (dict): Results from the run_all_dormant_identification_checks function
        df (pd.DataFrame): Original dataframe to be enhanced with dormant results

    Returns:
        pd.DataFrame: Enhanced dataframe with dormant flags ready for compliance checks
    """
    # Create a copy of the original dataframe to avoid modifying it
    enhanced_df = df.copy()

    # Initialize needed flags if they don't exist
    required_flags = [
        'Expected_Account_Dormant',
        'Expected_Requires_Article_3_Process',
        'Expected_Transfer_to_CB_Due',
        'Bank_Contact_Attempted_Post_Dormancy_Trigger'
    ]

    for flag_col in required_flags:
        if flag_col not in enhanced_df.columns:
            enhanced_df[flag_col] = False

    # Process each dormant agent result and set appropriate flags

    # 1. Accounts flagged dormant from any dormant check
    dormant_accounts = set()

    # Collect all identified dormant accounts from various checks
    for check_key, check_result in dormant_results.items():
        if isinstance(check_result, dict) and check_result.get('status') == 'dormant' and 'df' in check_result:
             for account_id in check_result['df']['Account_ID']:
                    dormant_accounts.add(account_id)


    # Set Expected_Account_Dormant flag for all identified dormant accounts
    enhanced_df.loc[
        enhanced_df['Account_ID'].isin(list(dormant_accounts)), 'Expected_Account_Dormant'] = True

    # 2. Accounts requiring Article 3 process
    if 'art3_process_needed' in dormant_results and isinstance(dormant_results['art3_process_needed'], dict) and 'df' in dormant_results['art3_process_needed']:
        art3_df = dormant_results['art3_process_needed']['df']
        if not art3_df.empty and 'Account_ID' in art3_df.columns:
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(art3_df['Account_ID']), 'Expected_Requires_Article_3_Process'] = True

    # 3. Accounts eligible for CB transfer
    if 'eligible_for_cb_transfer' in dormant_results and isinstance(dormant_results['eligible_for_cb_transfer'], dict) and 'df' in dormant_results['eligible_for_cb_transfer']:
        cb_transfer_df = dormant_results['eligible_for_cb_transfer']['df']
        if not cb_transfer_df.empty and 'Account_ID' in cb_transfer_df.columns:
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(cb_transfer_df['Account_ID']), 'Expected_Transfer_to_CB_Due'] = True

    # 4. Add appropriate contact flags based on contact attempts needed
    if 'proactive_contact_needed' in dormant_results and isinstance(dormant_results['proactive_contact_needed'], dict) and 'df' in dormant_results['proactive_contact_needed']:
        contact_df = dormant_results['proactive_contact_needed']['df']
        if not contact_df.empty and 'Account_ID' in contact_df.columns:
            # These accounts need contact but haven't been contacted yet
            enhanced_df.loc[enhanced_df['Account_ID'].isin(
                contact_df['Account_ID']), 'Bank_Contact_Attempted_Post_Dormancy_Trigger'] = False

    # 5. Process high value dormant accounts for special handling
    if 'high_value_dormant' in dormant_results and isinstance(dormant_results['high_value_dormant'], dict) and 'df' in dormant_results['high_value_dormant']:
        high_value_df = dormant_results['high_value_dormant']['df']
        if not high_value_df.empty and 'Account_ID' in high_value_df.columns:
            # Mark high value accounts for prioritized compliance handling
            if 'High_Value_Dormant' not in enhanced_df.columns:
                enhanced_df['High_Value_Dormant'] = False
            enhanced_df.loc[enhanced_df['Account_ID'].isin(high_value_df['Account_ID']), 'High_Value_Dormant'] = True

    # 6. Process dormant-to-active transitions
    if 'dormant_to_active' in dormant_results and isinstance(dormant_results['dormant_to_active'], dict) and 'df' in dormant_results['dormant_to_active']:
        reactivated_df = dormant_results['dormant_to_active']['df']
        if not reactivated_df.empty and 'Account_ID' in reactivated_df.columns:
            # These accounts should not be flagged as dormant anymore
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(reactivated_df['Account_ID']), 'Expected_Account_Dormant'] = False
            # They should also be removed from any CB transfer list
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(reactivated_df['Account_ID']), 'Expected_Transfer_to_CB_Due'] = False

    return enhanced_df


def apply_dormant_flags_to_dataframe(df, dormant_agent_results):
    """
    Apply dormant identification flags directly to the dataframe.
    This is a simpler alternative to the full process_dormant_results_for_compliance.

    Args:
        df (pd.DataFrame): Original dataframe
        dormant_agent_results (dict): Results from an individual dormant agent check

    Returns:
        pd.DataFrame: DataFrame with dormant flags applied
    """
    if not dormant_agent_results or 'df' not in dormant_agent_results or dormant_agent_results['df'].empty:
        return df

    # Make a copy to avoid modifying the original
    modified_df = df.copy()

    # Get the accounts identified by the dormant agent
    dormant_accounts_df = dormant_agent_results['df']

    if 'Account_ID' not in dormant_accounts_df.columns:
        return modified_df

    # Create or update the Expected_Account_Dormant flag
    if 'Expected_Account_Dormant' not in modified_df.columns:
        modified_df['Expected_Account_Dormant'] = False

    # Set the flag to True for identified accounts
    modified_df.loc[
        modified_df['Account_ID'].isin(dormant_accounts_df['Account_ID']), 'Expected_Account_Dormant'] = True

    return modified_df


def run_dormant_then_compliance(df, report_date, llm_client=None, config=None, flags_history_df=None):
    """
    Comprehensive function that runs dormant identification first and then compliance checks.

    Args:
        df (pd.DataFrame): Original data
        report_date: Reference date for dormancy calculations
        llm_client: The language model client.
        config: Configuration object.
        flags_history_df: Historical flags data for transitions

    Returns:
        tuple: (dormant_results, compliance_results, enhanced_df)
    """

    # Step 1: Run all dormant identification checks
    if isinstance(report_date, str):
        # Convert string date to datetime if needed
        report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
    else:
        report_date_obj = report_date

    dormant_orchestrator = DormantAccountOrchestrator(llm_client, config)
    dormant_results = {}
    for index, row in df.iterrows():
        dormant_results[row['Account_ID']] = dormant_orchestrator.run_all_agents(row.to_dict())


    # Step 2: Process results to enhance the dataframe with flags
    enhanced_df = process_dormant_results_for_compliance(dormant_results, df)

    # Step 3: Run compliance checks using the enhanced data
    compliance_orchestrator = ComplianceOrchestrator()
    compliance_results = {}

    for index, row in enhanced_df.iterrows():
        account_id = row['Account_ID']
        account_dormancy_results = dormant_results.get(account_id, {})

        # Instantiate all compliance agents
        agents = {agent_name: agent_class(llm_client, config) for agent_name, agent_class in compliance_orchestrator.AGENT_CLASS_MAP.items()}

        results = {}
        for agent_name, agent in agents.items():
            results[agent_name] = agent.execute(row.to_dict(), account_dormancy_results)
        compliance_results[account_id] = results


    return dormant_results, compliance_results, enhanced_df