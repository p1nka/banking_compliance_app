import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def check_safe_deposit(df, threshold_date):
    """
    Detects dormant safe deposit boxes according to UAE regulations:
    - Charges unpaid for more than 3 years
    - No response from tenant after bank contact attempts

    Args:
        df: DataFrame containing account data
        threshold_date: Date threshold (3 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'SDB_Charges_Outstanding', 'Date_SDB_Charges_Became_Outstanding',
            'SDB_Tenant_Communication_Received', 'Bank_Contact_Attempted_Post_Dormancy_Trigger'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Safe Deposit check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Date_SDB_Charges_Became_Outstanding']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter safe deposit boxes with outstanding charges > 3 years
        data = df[
            # Account type is Safe Deposit Box
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            # Has outstanding charges
            (df['SDB_Charges_Outstanding'].astype(float) > 0) &
            # Date when charges became outstanding is more than 3 years ago
            (df['Date_SDB_Charges_Became_Outstanding'].notna()) &
            (df['Date_SDB_Charges_Became_Outstanding'] < threshold_date) &
            # Bank contact was attempted
            (df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                ['yes', 'true', '1', 'y'])) &
            # No communication received from tenant
            (~df['SDB_Tenant_Communication_Received'].astype(str).str.lower().isin(['yes', 'true', '1', 'y']))
            ]

        count = len(data)
        desc = f"Safe Deposit boxes dormant (>3y unpaid fees): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit check: {e})"


def check_investment_inactivity(df, threshold_date):
    """
    Detects dormant investment accounts according to UAE regulations:
    - 3+ years from maturity/redemption date
    - No customer communication

    Args:
        df: DataFrame containing account data
        threshold_date: Date threshold (3 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'Inv_Maturity_Redemption_Date', 'Date_Last_Customer_Communication_Any_Type'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Investment check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Inv_Maturity_Redemption_Date', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter investment accounts dormant as per UAE regulation
        data = df[
            # Account type is Investment
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            # Has maturity/redemption date > 3 years ago
            (df['Inv_Maturity_Redemption_Date'].notna()) &
            (df['Inv_Maturity_Redemption_Date'] < threshold_date) &
            # Either no customer communication at all or last communication > 3 years ago
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) |
             (df['Date_Last_Customer_Communication_Any_Type'] < threshold_date))
            ]

        count = len(data)
        desc = f"Investment accounts dormant (>3y from maturity): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment inactivity check: {e})"


def check_fixed_deposit_inactivity(df, threshold_date):
    """
    Detects dormant fixed deposit accounts according to UAE regulations:
    - Matured deposits not renewed or claimed for 3+ years
    - No auto-renewal clause
    - No customer communication after maturity

    Args:
        df: DataFrame containing account data
        threshold_date: Date threshold (3 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'FTD_Maturity_Date', 'FTD_Auto_Renewal',
            'Date_Last_FTD_Renewal_Claim_Request', 'Date_Last_Customer_Communication_Any_Type'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Fixed Deposit check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['FTD_Maturity_Date', 'Date_Last_FTD_Renewal_Claim_Request',
                     'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter fixed deposit accounts dormant as per UAE regulation
        data = df[
            # Account type is Fixed Deposit
            (df['Account_Type'].astype(str).str.contains("Fixed|Term", case=False, na=False)) &
            # Has maturity date > 3 years ago
            (df['FTD_Maturity_Date'].notna()) &
            (df['FTD_Maturity_Date'] < threshold_date) &
            # No auto-renewal clause
            (~df['FTD_Auto_Renewal'].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])) &
            # Either no renewal/claim request at all or last request > 3 years ago
            ((df['Date_Last_FTD_Renewal_Claim_Request'].isna()) |
             (df['Date_Last_FTD_Renewal_Claim_Request'] < threshold_date)) &
            # Either no customer communication at all or last communication > 3 years ago
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) |
             (df['Date_Last_Customer_Communication_Any_Type'] < threshold_date))
            ]

        count = len(data)
        desc = f"Fixed deposit accounts dormant (>3y after maturity): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit inactivity check: {e})"


def check_demand_deposit_inactivity(df, threshold_date):
    """
    Detects dormant demand deposit accounts according to UAE regulations:
    - Current/Savings/Call accounts with 3+ years of inactivity
    - No customer-initiated activity
    - No customer communication

    Args:
        df: DataFrame containing account data
        threshold_date: Date threshold (3 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
            'Customer_Has_Active_Liability_Account'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Demand Deposit check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter demand deposit accounts dormant as per UAE regulation
        data = df[
            # Account type is Current, Savings or Call
            (df['Account_Type'].astype(str).str.contains("Current|Saving|Call", case=False, na=False)) &
            # Last customer activity > 3 years ago
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date) &
            # Either no customer communication at all or last communication > 3 years ago
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) |
             (df['Date_Last_Customer_Communication_Any_Type'] < threshold_date)) &
            # Customer does not have other active accounts with the bank
            (~df['Customer_Has_Active_Liability_Account'].astype(str).str.lower().isin(['yes', 'true', '1', 'y']))
            ]

        count = len(data)
        desc = f"Demand deposit accounts dormant (>3y inactivity): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Demand Deposit inactivity check: {e})"


def check_bankers_cheques(df, one_year_threshold):
    """
    Detects unclaimed bankers cheques, drafts, and cashier orders according to UAE regulations:
    - Unclaimed for 1+ year from issuance
    - No claim by beneficiary or customer

    Args:
        df: DataFrame containing account data
        one_year_threshold: Date threshold (1 year prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'Unclaimed_Item_Trigger_Date', 'Unclaimed_Item_Amount'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Bankers Cheques check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Unclaimed_Item_Trigger_Date']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter unclaimed bankers cheques as per UAE regulation
        data = df[
            # Account type is Banker's Cheque, Draft, or Cashier's Order
            (df['Account_Type'].astype(str).str.contains("Cheque|Draft|Order|Cashier", case=False, na=False)) &
            # Issue date > 1 year ago
            (df['Unclaimed_Item_Trigger_Date'].notna()) &
            (df['Unclaimed_Item_Trigger_Date'] < one_year_threshold) &
            # Has amount
            (df['Unclaimed_Item_Amount'].notna()) &
            (df['Unclaimed_Item_Amount'] > 0)
            ]

        count = len(data)
        desc = f"Unclaimed bankers cheques/drafts/orders (>1y): {count} instruments"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Bankers Cheques check: {e})"


def check_transfer_to_central_bank(df, five_year_threshold):
    """
    Identifies dormant accounts eligible for transfer to the Central Bank as per UAE regulations:
    - Dormant for 5+ years from last activity
    - No other active account with the bank
    - Current address unknown

    Args:
        df: DataFrame containing account data
        five_year_threshold: Date threshold (5 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
            'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
            'Customer_Has_Litigation_Regulatory_Reqs'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Central Bank Transfer check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter accounts eligible for Central Bank transfer
        data = df[
            # Not including cheques/drafts - they have their own process
            (~df['Account_Type'].astype(str).str.contains("Cheque|Draft|Order|Cashier", case=False, na=False)) &
            # Last activity > 5 years ago
            ((df['Date_Last_Cust_Initiated_Activity'].isna()) |
             (df['Date_Last_Cust_Initiated_Activity'] < five_year_threshold)) &
            # Either no customer communication at all or last communication > 5 years ago
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) |
             (df['Date_Last_Customer_Communication_Any_Type'] < five_year_threshold)) &
            # Current address not known
            (~df['Customer_Address_Known'].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])) &
            # No other active accounts with the bank
            (~df['Customer_Has_Active_Liability_Account'].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])) &
            # No litigation or regulatory requirements that would prevent transfer
            (~df['Customer_Has_Litigation_Regulatory_Reqs'].astype(str).str.lower().isin(['yes', 'true', '1', 'y']))
            ]

        count = len(data)
        desc = f"Accounts eligible for transfer to Central Bank (>5y dormant): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Central Bank Transfer check: {e})"


def check_art3_process_required(df, threshold_date):
    """
    Identifies accounts that have become dormant but still need to go through
    the required actions as per Article 3 of the UAE regulations:
    - Attempts to contact the customer
    - 3-month waiting period
    - Transfer to dormant accounts ledger

    Args:
        df: DataFrame containing account data
        threshold_date: Date threshold (3 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
            'Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Expected_Account_Dormant'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Article 3 Process check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Filter accounts that need Article 3 process
        data = df[
            # Account is flagged as dormant
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1', 'y'])) &
            # Last activity > 3 years ago
            ((df['Date_Last_Cust_Initiated_Activity'] < threshold_date) |
             (df['Date_Last_Cust_Initiated_Activity'].isna())) &
            # Either no customer communication at all or last communication > 3 years ago
            ((df['Date_Last_Customer_Communication_Any_Type'] < threshold_date) |
             (df['Date_Last_Customer_Communication_Any_Type'].isna())) &
            # No contact attempt has been made
            (~df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                ['yes', 'true', '1', 'y']))
            ]

        count = len(data)
        desc = f"Accounts requiring Article 3 process: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Article 3 Process check: {e})"


def check_contact_attempts_needed(df, threshold_date):
    """
    Identifies accounts where contact attempts are needed according to UAE regulations.

    Args:
        df: DataFrame containing account data
        threshold_date: Date threshold (3 years prior to report date)

    Returns:
        Filtered DataFrame, count, and description
    """
    try:
        required_columns = [
            'Account_ID', 'Customer_ID', 'Account_Type',
            'Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
            'Bank_Contact_Attempted_Post_Dormancy_Trigger'
        ]

        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Contact Attempts check)"

        # Convert date columns to datetime if they aren't already
        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Calculate when the account may become dormant (2.5 years inactivity)
        early_warning_threshold = threshold_date + timedelta(days=180)  # 6 months early warning

        # Filter accounts nearing dormancy where contact attempts should be made
        data = df[
            # Last activity between 2.5 and 3 years ago
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < early_warning_threshold) &
            (df['Date_Last_Cust_Initiated_Activity'] >= threshold_date) &
            # Last communication between 2.5 and 3 years ago or none
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) |
             ((df['Date_Last_Customer_Communication_Any_Type'] < early_warning_threshold) &
              (df['Date_Last_Customer_Communication_Any_Type'] >= threshold_date))) &
            # No contact attempt has been made yet
            (~df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                ['yes', 'true', '1', 'y']))
            ]

        count = len(data)
        desc = f"Accounts nearing dormancy needing contact attempts: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Contact Attempts check: {e})"


def run_all_dormant_checks(df, report_date=None):
    """
    Run all dormant account detection checks as per UAE Central Bank regulations.

    Args:
        df (pandas.DataFrame): The account data
        report_date (datetime, optional): Reference date for calculations, defaults to today

    Returns:
        dict: Dictionary containing results from all dormant checks
    """
    if report_date is None:
        report_date = datetime.now()

    # Calculate threshold dates based on UAE Central Bank rules
    three_year_threshold = report_date - timedelta(days=3 * 365)  # 3 years for standard dormancy
    one_year_threshold = report_date - timedelta(days=365)  # 1 year for banker's cheques
    five_year_threshold = report_date - timedelta(days=5 * 365)  # 5 years for Central Bank transfer

    results = {
        "total_accounts": len(df),
        "sd": {"df": None, "count": 0, "desc": ""},  # Safe Deposit
        "inv": {"df": None, "count": 0, "desc": ""},  # Investment Accounts
        "fd": {"df": None, "count": 0, "desc": ""},  # Fixed Deposits
        "dd": {"df": None, "count": 0, "desc": ""},  # Demand Deposits (renamed from gen)
        "chq": {"df": None, "count": 0, "desc": ""},  # Bankers Cheques
        "cb": {"df": None, "count": 0, "desc": ""},  # Central Bank Transfer
        "art3": {"df": None, "count": 0, "desc": ""},  # Article 3 Process Required
        "con": {"df": None, "count": 0, "desc": ""}  # Contact Attempts Needed
    }

    # Run each check
    results["sd"]["df"], results["sd"]["count"], results["sd"]["desc"] = check_safe_deposit(df, three_year_threshold)
    results["inv"]["df"], results["inv"]["count"], results["inv"]["desc"] = check_investment_inactivity(df,
                                                                                                        three_year_threshold)
    results["fd"]["df"], results["fd"]["count"], results["fd"]["desc"] = check_fixed_deposit_inactivity(df,
                                                                                                        three_year_threshold)
    results["dd"]["df"], results["dd"]["count"], results["dd"]["desc"] = check_demand_deposit_inactivity(df,
                                                                                                         three_year_threshold)
    results["chq"]["df"], results["chq"]["count"], results["chq"]["desc"] = check_bankers_cheques(df,
                                                                                                  one_year_threshold)
    results["cb"]["df"], results["cb"]["count"], results["cb"]["desc"] = check_transfer_to_central_bank(df,
                                                                                                        five_year_threshold)
    results["art3"]["df"], results["art3"]["count"], results["art3"]["desc"] = check_art3_process_required(df,
                                                                                                           three_year_threshold)
    results["con"]["df"], results["con"]["count"], results["con"]["desc"] = check_contact_attempts_needed(df,
                                                                                                          three_year_threshold)

    # Calculate overall statistics
    total_dormant = (
            results["sd"]["count"] + results["inv"]["count"] +
            results["fd"]["count"] + results["dd"]["count"]
    )

    results["statistics"] = {
        "total_dormant": total_dormant,
        "dormant_percentage": round((total_dormant / results["total_accounts"]) * 100, 2) if results[
                                                                                                 "total_accounts"] > 0 else 0,
        "cb_transfer_percentage": round((results["cb"]["count"] / total_dormant) * 100, 2) if total_dormant > 0 else 0,
        "unclaimed_instruments": results["chq"]["count"],
        "report_date": report_date.strftime("%Y-%m-%d")
    }

    return results