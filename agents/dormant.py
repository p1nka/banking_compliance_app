import pandas as pd
from datetime import datetime, timedelta


def check_safe_deposit(df, threshold_date_3y, threshold_date_5y=None):
    """
    Detects safe deposit accounts inactive over threshold with no contact attempts.

    Args:
        df (pandas.DataFrame): Account data
        threshold_date_3y (datetime): 3-year threshold for initial dormancy
        threshold_date_5y (datetime, optional): 5-year threshold for central bank transfer

    Returns:
        tuple: (filtered DataFrame, count, description, transfer_df)
    """
    try:
        # Ensure columns exist and handle potential None/NaN in string comparisons
        if not all(col in df.columns for col in
                   ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt',
                    'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Safe Deposit check)", pd.DataFrame()

        # Initial dormancy check (3 years)
        data = df[
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            (df['Last_Transaction_Date'].notna()) &  # Ensure date is not NaT
            (df['Last_Transaction_Date'] < threshold_date_3y) &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &  # Handle NaNs by converting to str
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]

        # Check for accounts ready for central bank transfer (5 years)
        transfer_df = pd.DataFrame()
        if threshold_date_5y is not None:
            transfer_df = df[
                (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
                (df['Last_Transaction_Date'].notna()) &
                (df['Last_Transaction_Date'] < threshold_date_5y) &
                (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
                (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
                (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
                ]

        count = len(data)
        transfer_count = len(transfer_df)
        desc = f"Safe Deposit without contact attempts (>3y): {count} accounts, ({transfer_count} eligible for Central Bank transfer)"
        return data, count, desc, transfer_df
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit check: {e})", pd.DataFrame()


def check_investment_inactivity(df, threshold_date_3y, threshold_date_5y=None):
    """
    Detects investment accounts inactive over threshold with no contact attempts.

    Args:
        df (pandas.DataFrame): Account data
        threshold_date_3y (datetime): 3-year threshold for initial dormancy
        threshold_date_5y (datetime, optional): 5-year threshold for central bank transfer

    Returns:
        tuple: (filtered DataFrame, count, description, transfer_df)
    """
    try:
        if not all(col in df.columns for col in
                   ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt',
                    'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Investment check)", pd.DataFrame()

        # Initial dormancy check (3 years)
        data = df[
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date_3y) &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]

        # Check for accounts ready for central bank transfer (5 years)
        transfer_df = pd.DataFrame()
        if threshold_date_5y is not None:
            transfer_df = df[
                (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
                (df['Last_Transaction_Date'].notna()) &
                (df['Last_Transaction_Date'] < threshold_date_5y) &
                (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
                (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
                (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
                ]

        count = len(data)
        transfer_count = len(transfer_df)
        desc = f"Investment accounts without activity or contact (>3y): {count} accounts, ({transfer_count} eligible for Central Bank transfer)"
        return data, count, desc, transfer_df
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment inactivity check: {e})", pd.DataFrame()


def check_fixed_deposit_inactivity(df, threshold_date_3y, threshold_date_5y=None):
    """
    Detects fixed deposit accounts inactive over threshold.

    Args:
        df (pandas.DataFrame): Account data
        threshold_date_3y (datetime): 3-year threshold for initial dormancy
        threshold_date_5y (datetime, optional): 5-year threshold for central bank transfer

    Returns:
        tuple: (filtered DataFrame, count, description, transfer_df)
    """
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Fixed Deposit check)", pd.DataFrame()

        # Initial dormancy check (3 years)
        data = df[
            (df['Account_Type'].astype(str).str.lower() == 'fixed deposit') &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date_3y)
            ]

        # Check for accounts ready for central bank transfer (5 years)
        transfer_df = pd.DataFrame()
        if threshold_date_5y is not None:
            transfer_df = df[
                (df['Account_Type'].astype(str).str.lower() == 'fixed deposit') &
                (df['Last_Transaction_Date'].notna()) &
                (df['Last_Transaction_Date'] < threshold_date_5y)
                ]

        count = len(data)
        transfer_count = len(transfer_df)
        desc = f"Fixed deposit accounts with no activity (>3y): {count} accounts, ({transfer_count} eligible for Central Bank transfer)"
        return data, count, desc, transfer_df
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit inactivity check: {e})", pd.DataFrame()


def check_general_inactivity(df, threshold_date_3y, threshold_date_5y=None):
    """
    Detects Savings/Call/Current accounts inactive over threshold.

    Args:
        df (pandas.DataFrame): Account data
        threshold_date_3y (datetime): 3-year threshold for initial dormancy
        threshold_date_5y (datetime, optional): 5-year threshold for central bank transfer

    Returns:
        tuple: (filtered DataFrame, count, description, transfer_df)
    """
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for General Inactivity check)", pd.DataFrame()

        # Initial dormancy check (3 years)
        data = df[
            (df['Account_Type'].astype(str).isin(["Savings", "Call", "Current"])) &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date_3y)
            ]

        # Check for accounts ready for central bank transfer (5 years)
        transfer_df = pd.DataFrame()
        if threshold_date_5y is not None:
            transfer_df = df[
                (df['Account_Type'].astype(str).isin(["Savings", "Call", "Current"])) &
                (df['Last_Transaction_Date'].notna()) &
                (df['Last_Transaction_Date'] < threshold_date_5y)
                ]

        count = len(data)
        transfer_count = len(transfer_df)
        desc = f"General accounts (Savings/Call/Current) with no activity (>3y): {count} accounts, ({transfer_count} eligible for Central Bank transfer)"
        return data, count, desc, transfer_df
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in General inactivity check: {e})", pd.DataFrame()


def check_unreachable_dormant(df):
    """
    Detects accounts marked dormant with no contact attempts.

    Args:
        df (pandas.DataFrame): Account data

    Returns:
        tuple: (filtered DataFrame, count, description, transfer_df)
    """
    try:
        if not all(col in df.columns for col in
                   ['Account_Status', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Unreachable Dormant check)", pd.DataFrame()

        data = df[
            (df['Account_Status'].astype(str).str.lower() == 'dormant') &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]
        count = len(data)
        desc = f"Unreachable accounts already marked dormant: {count} accounts"
        # For unreachable accounts, same df is returned for transfer consideration
        return data, count, desc, data
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unreachable dormant check: {e})", pd.DataFrame()


def convert_foreign_currencies(df):
    """
    Convert foreign currencies to AED as per CBUAE regulations.

    Args:
        df (pandas.DataFrame): Accounts data with balances to be converted

    Returns:
        pandas.DataFrame: DataFrame with balances converted to AED
    """
    try:
        # This is a placeholder for the actual conversion logic
        # In a real implementation, you would:
        # 1. Check if 'Currency' and 'Balance' columns exist
        # 2. Get current exchange rates from a reliable source
        # 3. Convert non-AED balances to AED

        if not all(col in df.columns for col in ['Currency', 'Balance']):
            return df

        # Make a copy to avoid SettingWithCopyWarning
        result_df = df.copy()

        # Identify non-AED currencies
        non_aed = result_df['Currency'] != 'AED'

        if non_aed.any():
            # In a real implementation, you would apply actual conversion rates
            # This is just a placeholder example
            result_df.loc[non_aed, 'Balance_AED'] = result_df.loc[non_aed, 'Balance']
            result_df.loc[~non_aed, 'Balance_AED'] = result_df.loc[~non_aed, 'Balance']
            result_df['Original_Currency'] = result_df['Currency']
            result_df['Currency'] = 'AED'

        return result_df
    except Exception as e:
        print(f"Error in currency conversion: {e}")
        return df


def run_all_dormant_checks(df, threshold_date_3y, threshold_date_5y=None):
    """
    Run all dormant account detection checks and return a consolidated result.

    Args:
        df (pandas.DataFrame): The account data
        threshold_date_3y (datetime): 3-year date threshold for initial dormancy
        threshold_date_5y (datetime, optional): 5-year date threshold for central bank transfer

    Returns:
        dict: Dictionary containing results from all dormant checks
    """
    results = {
        "total_accounts": len(df),
        "sd": {"df": None, "count": 0, "desc": "", "transfer_df": None},
        "inv": {"df": None, "count": 0, "desc": "", "transfer_df": None},
        "fd": {"df": None, "count": 0, "desc": "", "transfer_df": None},
        "gen": {"df": None, "count": 0, "desc": "", "transfer_df": None},
        "unr": {"df": None, "count": 0, "desc": "", "transfer_df": None}
    }

    # Run each check
    results["sd"]["df"], results["sd"]["count"], results["sd"]["desc"], results["sd"][
        "transfer_df"] = check_safe_deposit(
        df, threshold_date_3y, threshold_date_5y)

    results["inv"]["df"], results["inv"]["count"], results["inv"]["desc"], results["inv"][
        "transfer_df"] = check_investment_inactivity(
        df, threshold_date_3y, threshold_date_5y)

    results["fd"]["df"], results["fd"]["count"], results["fd"]["desc"], results["fd"][
        "transfer_df"] = check_fixed_deposit_inactivity(
        df, threshold_date_3y, threshold_date_5y)

    results["gen"]["df"], results["gen"]["count"], results["gen"]["desc"], results["gen"][
        "transfer_df"] = check_general_inactivity(
        df, threshold_date_3y, threshold_date_5y)

    results["unr"]["df"], results["unr"]["count"], results["unr"]["desc"], results["unr"][
        "transfer_df"] = check_unreachable_dormant(df)

    # If we have accounts to transfer, convert any foreign currencies to AED
    if threshold_date_5y is not None:
        for check_type in ["sd", "inv", "fd", "gen", "unr"]:
            if results[check_type]["transfer_df"] is not None and len(results[check_type]["transfer_df"]) > 0:
                results[check_type]["transfer_df"] = convert_foreign_currencies(results[check_type]["transfer_df"])

    return results


def prepare_central_bank_transfer(df):
    """
    Prepare accounts data for transfer to Central Bank.

    Args:
        df (pandas.DataFrame): Accounts data ready for transfer

    Returns:
        pandas.DataFrame: Formatted data for Central Bank reporting
    """
    try:
        # Create a copy to avoid modifying the original DataFrame
        cb_df = df.copy()

        # Add required fields for Central Bank reporting
        # These fields are based on the movement form shown in the regulation document
        cb_df['Transfer_Date'] = datetime.now().strftime('%Y-%m-%d')

        # Format according to BRF (Banking Return Forms) requirements
        # This is a placeholder for the actual formatting logic

        return cb_df
    except Exception as e:
        print(f"Error preparing Central Bank transfer data: {e}")
        return pd.DataFrame()


def identify_high_value_dormant(df, threshold=25000):
    """Identify dormant accounts with balance >= AED 25,000"""
    if 'Balance' not in df.columns:
        return pd.DataFrame(), 0, "Balance column missing"

    high_value = df[df['Balance'] >= threshold]
    return high_value, len(high_value), f"High-value dormant accounts (≥ AED {threshold}): {len(high_value)}"


def detect_post_dormancy_transactions(df, dormant_accounts_df):
    """Detect transactions occurring after dormancy classification"""
    if not all(col in df.columns for col in ['Account_Number', 'Transaction_Date']):
        return pd.DataFrame(), 0, "Required columns missing"

    # Get list of dormant account numbers
    dormant_numbers = dormant_accounts_df['Account_Number'].unique()

    # Find transactions for these accounts after dormancy date
    # Note: Would need a 'Dormancy_Date' field to implement properly
    # This is a conceptual placeholder
    post_dormancy = df[df['Account_Number'].isin(dormant_numbers)]

    return post_dormancy, len(post_dormancy), f"Accounts with post-dormancy transactions: {len(post_dormancy)}"


def check_customer_notifications(df):
    """Track percentage of dormant accounts where customer was notified"""
    if 'Customer_Notified' not in df.columns:
        return 0, "Customer notification tracking not available"

    notified_count = df['Customer_Notified'].sum()
    percentage = (notified_count / len(df)) * 100 if len(df) > 0 else 0

    return percentage, f"Customer notification rate: {percentage:.2f}%"


def track_dormant_to_active(df, history_df):
    """Track accounts reactivated through valid customer contact"""
    if not all(col in history_df.columns for col in ['Account_Number', 'Status_Change', 'Change_Date']):
        return pd.DataFrame(), 0, "Required columns missing from history"

    # Find accounts that changed from dormant to active
    reactivated = history_df[
        (history_df['Status_Change'] == 'Dormant to Active')
    ]

    return reactivated, len(reactivated), f"Accounts reactivated: {len(reactivated)}"


def track_manual_overrides(df, audit_df):
    """Track manually overridden dormancy classifications"""
    if not all(col in audit_df.columns for col in ['Account_Number', 'Override_Type', 'Approver']):
        return pd.DataFrame(), 0, "Required audit columns missing"

    # Find manual overrides
    overrides = audit_df[
        (audit_df['Override_Type'] == 'Dormancy Classification')
    ]

    return overrides, len(overrides), f"Manual dormancy overrides: {len(overrides)}"


def generate_cbuae_report(results, reporting_date):
    """Generate regulatory report for CBUAE"""
    # Create a report dataframe with required metrics
    report = {
        'Report_Date': reporting_date,
        'Total_Dormant_Accounts': sum(results[k]['count'] for k in ['sd', 'inv', 'fd', 'gen', 'unr']),
        'Total_Dormant_Balance': 0,  # Would need to calculate from account balances
        'High_Value_Accounts': 0,  # Would need high-value count
        'Accounts_Eligible_For_Transfer': sum(len(results[k]['transfer_df']) for k in ['sd', 'inv', 'fd', 'gen', 'unr']
                                              if results[k]['transfer_df'] is not None),
        'Submission_Days': 0  # Would track days from trigger to submission
    }

    return pd.DataFrame([report])


def calculate_dormancy_percentage(total_dormant, total_accounts):
    """Calculate dormant accounts as percentage of total customer base"""
    return (total_dormant / total_accounts) * 100 if total_accounts > 0 else 0