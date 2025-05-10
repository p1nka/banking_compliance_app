import pandas as pd
from datetime import datetime, timedelta


def check_safe_deposit(df, threshold_date):
    """Detects safe deposit accounts inactive over threshold with no contact attempts."""
    try:
        # Ensure columns exist and handle potential None/NaN in string comparisons
        if not all(col in df.columns for col in
                   ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt',
                    'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Safe Deposit check)"

        data = df[
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            (df['Last_Transaction_Date'].notna()) &  # Ensure date is not NaT
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &  # Handle NaNs by converting to str
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]
        count = len(data)
        desc = f"Safe Deposit without contact attempts (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit check: {e})"


def check_investment_inactivity(df, threshold_date):
    """Detects investment accounts inactive over threshold with no contact attempts."""
    try:
        if not all(col in df.columns for col in
                   ['Account_Type', 'Last_Transaction_Date', 'Email_Contact_Attempt', 'SMS_Contact_Attempt',
                    'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Investment check)"

        data = df[
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]
        count = len(data)
        desc = f"Investment accounts without activity or contact (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment inactivity check: {e})"


def check_fixed_deposit_inactivity(df, threshold_date):
    """Detects fixed deposit accounts inactive over threshold."""
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Fixed Deposit check)"

        data = df[
            (df['Account_Type'].astype(str).str.lower() == 'fixed deposit') &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date)
            ]
        count = len(data)
        desc = f"Fixed deposit accounts with no activity (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit inactivity check: {e})"


def check_general_inactivity(df, threshold_date):
    """Detects Savings/Call/Current accounts inactive over threshold."""
    try:
        if not all(col in df.columns for col in ['Account_Type', 'Last_Transaction_Date']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for General Inactivity check)"

        data = df[
            (df['Account_Type'].astype(str).isin(["Savings", "Call", "Current"])) &
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date)
            ]
        count = len(data)
        desc = f"General accounts (Savings/Call/Current) with no activity (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in General inactivity check: {e})"


def check_unreachable_dormant(df):
    """Detects accounts marked dormant with no contact attempts."""
    try:
        if not all(col in df.columns for col in
                   ['Account_Status', 'Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Unreachable Dormant check)"

        data = df[
            (df['Account_Status'].astype(str).str.lower() == 'dormant') &
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') &
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]
        count = len(data)
        desc = f"Unreachable accounts already marked dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unreachable dormant check: {e})"


def run_all_dormant_checks(df, threshold_date):
    """
    Run all dormant account detection checks and return a consolidated result.

    Args:
        df (pandas.DataFrame): The account data
        threshold_date (datetime): Date threshold for inactivity

    Returns:
        dict: Dictionary containing results from all dormant checks
    """
    results = {
        "total_accounts": len(df),
        "sd": {"df": None, "count": 0, "desc": ""},
        "inv": {"df": None, "count": 0, "desc": ""},
        "fd": {"df": None, "count": 0, "desc": ""},
        "gen": {"df": None, "count": 0, "desc": ""},
        "unr": {"df": None, "count": 0, "desc": ""}
    }

    # Run each check
    results["sd"]["df"], results["sd"]["count"], results["sd"]["desc"] = check_safe_deposit(df, threshold_date)
    results["inv"]["df"], results["inv"]["count"], results["inv"]["desc"] = check_investment_inactivity(df,
                                                                                                        threshold_date)
    results["fd"]["df"], results["fd"]["count"], results["fd"]["desc"] = check_fixed_deposit_inactivity(df,
                                                                                                        threshold_date)
    results["gen"]["df"], results["gen"]["count"], results["gen"]["desc"] = check_general_inactivity(df, threshold_date)
    results["unr"]["df"], results["unr"]["count"], results["unr"]["desc"] = check_unreachable_dormant(df)

    return results