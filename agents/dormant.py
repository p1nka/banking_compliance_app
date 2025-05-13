import pandas as pd
from datetime import datetime, timedelta


def check_safe_deposit(df, threshold_date):
    """Detects safe deposit boxes with outstanding charges for CBUAE compliance."""
    try:
        # Check for necessary columns per CBUAE regulation
        required_cols = ['Account_Type', 'SDB_Charges_Outstanding', 'Date_SDB_Charges_Became_Outstanding',
                         'SDB_Tenant_Communication_Received']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Per Article 2(6) of CBUAE regulation: SDB is dormant if charges are outstanding for >3 years
        # and no communication from tenant
        data = df[
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            (df['SDB_Charges_Outstanding'].notna()) &
            (df['SDB_Charges_Outstanding'] > 0) &
            (df['Date_SDB_Charges_Became_Outstanding'].notna()) &
            (df['Date_SDB_Charges_Became_Outstanding'] < threshold_date) &
            (df['SDB_Tenant_Communication_Received'].astype(str).str.lower() == 'no')
            ]

        count = len(data)
        desc = f"Safe Deposit Box with outstanding charges (>3y): {count} boxes"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit check: {e})"


def check_investment_inactivity(df, threshold_date):
    """Detects investment accounts inactive over threshold with no contact attempts per CBUAE Article 2(3)."""
    try:
        # Check for necessary columns per CBUAE regulation
        required_cols = ['Account_Type', 'Inv_Maturity_Redemption_Date',
                         'Date_Last_Customer_Communication_Any_Type']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Per Article 2(3) of CBUAE regulation: Investment account is dormant if
        # no contact for 3 years from maturity/redemption
        data = df[
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            (df['Inv_Maturity_Redemption_Date'].notna()) &
            (df['Inv_Maturity_Redemption_Date'] < threshold_date) &
            (
                    (df['Date_Last_Customer_Communication_Any_Type'].isna()) |
                    (df['Date_Last_Customer_Communication_Any_Type'] < df['Inv_Maturity_Redemption_Date'])
            )
            ]

        count = len(data)
        desc = f"Investment accounts with no contact since maturity/redemption (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment inactivity check: {e})"


def check_fixed_deposit_inactivity(df, threshold_date):
    """Detects fixed deposit accounts inactive over threshold per CBUAE Article 2(2)."""
    try:
        # Check for necessary columns per CBUAE regulation
        required_cols = ['Account_Type', 'FTD_Maturity_Date', 'FTD_Auto_Renewal',
                         'Date_Last_FTD_Renewal_Claim_Request']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Per Article 2(2) of CBUAE regulation:
        # Case 1: No auto-renewal, dormant if 3 years after maturity and no renewal/claim
        # Case 2: Has auto-renewal, dormant if no communication for 3 years from first maturity

        # Case 1
        no_auto_renew = df[
            (df['Account_Type'].astype(str).str.lower() == 'fixed deposit') &
            (df['FTD_Auto_Renewal'].astype(str).str.lower() == 'no') &
            (df['FTD_Maturity_Date'].notna()) &
            (df['FTD_Maturity_Date'] < threshold_date) &
            (
                    (df['Date_Last_FTD_Renewal_Claim_Request'].isna()) |
                    (df['Date_Last_FTD_Renewal_Claim_Request'] < df['FTD_Maturity_Date'])
            )
            ]

        # Case 2
        auto_renew = df[
            (df['Account_Type'].astype(str).str.lower() == 'fixed deposit') &
            (df['FTD_Auto_Renewal'].astype(str).str.lower() == 'yes') &
            (df['FTD_Maturity_Date'].notna()) &
            (df['FTD_Maturity_Date'] < threshold_date) &
            (
                    (df['Date_Last_Customer_Communication_Any_Type'].isna()) |
                    (df['Date_Last_Customer_Communication_Any_Type'] < df['FTD_Maturity_Date'])
            )
            ]

        # Combine the results
        data = pd.concat([no_auto_renew, auto_renew]).drop_duplicates()

        count = len(data)
        desc = f"Fixed deposit accounts meeting CBUAE dormancy criteria: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit inactivity check: {e})"


def check_general_inactivity(df, threshold_date):
    """Detects Savings/Call/Current accounts inactive over threshold per CBUAE Article 2(1)."""
    try:
        # Check for necessary columns per CBUAE regulation
        required_cols = ['Account_Type', 'Date_Last_Cust_Initiated_Activity']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Per Article 2(1) of CBUAE regulation: Account is dormant if no customer transaction for 3 years
        data = df[
            (df['Account_Type'].astype(str).str.lower().isin(["savings", "call", "current"])) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date)
            ]

        count = len(data)
        desc = f"General accounts (Savings/Call/Current) with no customer activity (>3y): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in General inactivity check: {e})"


def check_unreachable_dormant(df):
    """Detects accounts marked dormant with no contact attempts per CBUAE Article 2(5)."""
    try:
        # Check for necessary columns per CBUAE regulation
        required_cols = ['Expected_Account_Dormant', 'Customer_Address_Known',
                         'Customer_Has_Active_Liability_Account']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Per Article 2(5) of CBUAE regulation: Account is dormant if:
        # - Already marked as potentially dormant
        # - Current address is unknown
        # - Customer has no other active liability accounts
        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Customer_Address_Known'].astype(str).str.lower() == 'no') &
            (df['Customer_Has_Active_Liability_Account'].astype(str).str.lower() == 'no')
            ]

        count = len(data)
        desc = f"Unreachable accounts marked potentially dormant: {count} accounts"
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