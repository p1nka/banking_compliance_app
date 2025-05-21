"""
Connector Module for Dormant and Compliance Agents

This module provides functions that connect the outputs of dormant agents
with the inputs required by compliance agents. It handles data transformation,
aggregation, and flag propagation between agent systems.
"""

import pandas as pd
from datetime import datetime, timedelta


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
            enhanced_df[flag_col] = 'no'  # Default is no/false

    # Process each dormant agent result and set appropriate flags

    # 1. Accounts flagged dormant from any dormant check
    dormant_accounts = pd.DataFrame()

    # Collect all identified dormant accounts from various checks
    for check_key in ['sdb_dormant', 'investment_dormant', 'fixed_deposit_dormant',
                      'demand_deposit_dormant', 'unclaimed_instruments']:
        if check_key in dormant_results and 'df' in dormant_results[check_key]:
            check_df = dormant_results[check_key]['df']
            if not check_df.empty and 'Account_ID' in check_df.columns:
                dormant_accounts = pd.concat([dormant_accounts, check_df[['Account_ID']]])

    # Remove duplicates if an account was identified by multiple checks
    if not dormant_accounts.empty:
        dormant_accounts = dormant_accounts.drop_duplicates(subset=['Account_ID'])

        # Set Expected_Account_Dormant flag for all identified dormant accounts
        enhanced_df.loc[
            enhanced_df['Account_ID'].isin(dormant_accounts['Account_ID']), 'Expected_Account_Dormant'] = 'yes'

    # 2. Accounts requiring Article 3 process
    if 'art3_process_needed' in dormant_results and 'df' in dormant_results['art3_process_needed']:
        art3_df = dormant_results['art3_process_needed']['df']
        if not art3_df.empty and 'Account_ID' in art3_df.columns:
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(art3_df['Account_ID']), 'Expected_Requires_Article_3_Process'] = 'yes'

    # 3. Accounts eligible for CB transfer
    if 'eligible_for_cb_transfer' in dormant_results and 'df' in dormant_results['eligible_for_cb_transfer']:
        cb_transfer_df = dormant_results['eligible_for_cb_transfer']['df']
        if not cb_transfer_df.empty and 'Account_ID' in cb_transfer_df.columns:
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(cb_transfer_df['Account_ID']), 'Expected_Transfer_to_CB_Due'] = 'yes'

    # 4. Add appropriate contact flags based on contact attempts needed
    if 'proactive_contact_needed' in dormant_results and 'df' in dormant_results['proactive_contact_needed']:
        contact_df = dormant_results['proactive_contact_needed']['df']
        if not contact_df.empty and 'Account_ID' in contact_df.columns:
            # These accounts need contact but haven't been contacted yet
            enhanced_df.loc[enhanced_df['Account_ID'].isin(
                contact_df['Account_ID']), 'Bank_Contact_Attempted_Post_Dormancy_Trigger'] = 'no'

    # 5. Process high value dormant accounts for special handling
    if 'high_value_dormant' in dormant_results and 'df' in dormant_results['high_value_dormant']:
        high_value_df = dormant_results['high_value_dormant']['df']
        if not high_value_df.empty and 'Account_ID' in high_value_df.columns:
            # Mark high value accounts for prioritized compliance handling
            if 'High_Value_Dormant' not in enhanced_df.columns:
                enhanced_df['High_Value_Dormant'] = 'no'
            enhanced_df.loc[enhanced_df['Account_ID'].isin(high_value_df['Account_ID']), 'High_Value_Dormant'] = 'yes'

    # 6. Process dormant-to-active transitions
    if 'dormant_to_active' in dormant_results and 'df' in dormant_results['dormant_to_active']:
        reactivated_df = dormant_results['dormant_to_active']['df']
        if not reactivated_df.empty and 'Account_ID' in reactivated_df.columns:
            # These accounts should not be flagged as dormant anymore
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(reactivated_df['Account_ID']), 'Expected_Account_Dormant'] = 'no'
            # They should also be removed from any CB transfer list
            enhanced_df.loc[
                enhanced_df['Account_ID'].isin(reactivated_df['Account_ID']), 'Expected_Transfer_to_CB_Due'] = 'no'

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
        modified_df['Expected_Account_Dormant'] = 'no'

    # Set the flag to 'yes' for identified accounts
    modified_df.loc[
        modified_df['Account_ID'].isin(dormant_accounts_df['Account_ID']), 'Expected_Account_Dormant'] = 'yes'

    return modified_df


def run_dormant_then_compliance(df, report_date, flags_history_df=None, agent_name="ComplianceSystem"):
    """
    Comprehensive function that runs dormant identification first and then compliance checks.

    Args:
        df (pd.DataFrame): Original data
        report_date: Reference date for dormancy calculations
        flags_history_df: Historical flags data for transitions
        agent_name: Name for logging purposes

    Returns:
        tuple: (dormant_results, compliance_results, enhanced_df)
    """
    from agents.dormant import run_all_dormant_identification_checks
    from agents.compliance import run_all_compliance_checks
    from datetime import datetime, timedelta

    # Step 1: Run all dormant identification checks
    if isinstance(report_date, str):
        # Convert string date to datetime if needed
        report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
    else:
        report_date_obj = report_date

    dormant_results = run_all_dormant_identification_checks(
        df.copy(),
        report_date_str=report_date_obj.strftime("%Y-%m-%d"),
        dormant_flags_history_df=flags_history_df
    )

    # Step 2: Process results to enhance the dataframe with flags
    enhanced_df = process_dormant_results_for_compliance(dormant_results, df)

    # Step 3: Run compliance checks using the enhanced data
    # Calculate threshold dates for compliance
    general_threshold_date = report_date_obj - timedelta(days=3 * 365)  # 3 years for dormancy
    freeze_threshold_date = report_date_obj - timedelta(days=3 * 365)  # 3 years for statement freeze

    compliance_results = run_all_compliance_checks(
        enhanced_df.copy(),
        general_threshold_date=general_threshold_date,
        freeze_threshold_date=freeze_threshold_date,
        agent_name=agent_name
    )

    return dormant_results, compliance_results, enhanced_df