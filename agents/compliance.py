import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from database.connection import get_db_connection


def detect_incomplete_contact(df):
    """
    Detects accounts with incomplete contact attempts per UAE regulations.
    The regulation requires multiple channels for contact attempts.
    """
    try:
        if 'Bank_Contact_Attempted_Post_Dormancy_Trigger' not in df.columns:
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Incomplete Contact check)"

        # Find accounts where contact was not attempted or response not received
        data = df[
            (df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower() == 'no') |
            ((df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower() == 'yes') &
             (df['Customer_Responded_to_Bank_Contact'].astype(str).str.lower() == 'no'))
            ]
        count = len(data)
        desc = f"Accounts with incomplete contact attempts: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in contact attempt verification: {e})"


def detect_flag_candidates(df, threshold_date):
    """
    Detects accounts inactive over threshold (3 years for most account types per UAE regulation),
    not yet flagged dormant.
    """
    try:
        required_cols = ['Date_Last_Cust_Initiated_Activity', 'Expected_Account_Dormant', 'Account_Type']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Flag Candidate check)"

        # Convert date columns to datetime if they're not already
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'],
                                                                     errors='coerce')

        # Standard accounts are dormant after 3 years of inactivity
        standard_accounts = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date) &
            (df['Expected_Account_Dormant'].astype(str).str.lower() != 'yes') &
            (~df['Account_Type'].astype(str).str.lower().isin(['bankers_cheque', 'bank_draft', 'cashier_order']))
            ]

        # Bankers cheques, drafts, and cashiers orders are dormant after 1 year
        one_year_ago = datetime.now() - timedelta(days=365)
        payment_instruments = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < one_year_ago) &
            (df['Expected_Account_Dormant'].astype(str).str.lower() != 'yes') &
            (df['Account_Type'].astype(str).str.lower().isin(['bankers_cheque', 'bank_draft', 'cashier_order']))
            ]

        # Combine results
        data = pd.concat([standard_accounts, payment_instruments])
        count = len(data)
        desc = f"Accounts inactive over threshold, not yet flagged dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in dormant flag detection: {e})"


def detect_ledger_candidates(df):
    """
    Detects accounts marked dormant requiring Article 3 process (notification and 3-month waiting period).
    """
    try:
        required_cols = ['Expected_Account_Dormant', 'Expected_Requires_Article_3_Process',
                         'Date_Last_Bank_Contact_Attempt']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Ledger Candidate check)"

        # Get accounts that are dormant but haven't gone through Article 3 process
        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Expected_Requires_Article_3_Process'].astype(str).str.lower() == 'yes')
            ].copy()

        # Check if notification was sent over 3 months ago
        if not pd.api.types.is_datetime64_dtype(data['Date_Last_Bank_Contact_Attempt']):
            data['Date_Last_Bank_Contact_Attempt'] = pd.to_datetime(data['Date_Last_Bank_Contact_Attempt'],
                                                                    errors='coerce')

        three_months_ago = datetime.now() - timedelta(days=90)
        data = data[data['Date_Last_Bank_Contact_Attempt'] < three_months_ago]

        ids_in_ledger = []
        conn = get_db_connection()  # Uses cached connection
        if conn:  # Only attempt DB check if connection is available
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT account_id FROM dormant_ledger")
                    ids_in_ledger = [row[0] for row in cursor.fetchall()]
            except Exception as db_e:
                st.warning(f"Could not check dormant ledger table: {db_e}. Proceeding without filtering.")
                ids_in_ledger = []  # Ensure it's an empty list if check fails

        if ids_in_ledger:
            data = data[~data['Account_ID'].isin(ids_in_ledger)]

        count = len(data)
        desc = f"Dormant accounts requiring Article 3 process (after 3-month waiting period): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in ledger candidate detection: {e})"


def detect_freeze_candidates(df, threshold_date):
    """
    Detects dormant accounts inactive beyond freeze threshold that should have physical
    and electronic statement generation suppressed per Article 3.8 of UAE regulation.
    """
    try:
        required_cols = ['Date_Last_Cust_Initiated_Activity', 'Expected_Account_Dormant']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Freeze Candidate check)"

        # Convert date columns to datetime if they're not already
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'],
                                                                     errors='coerce')

        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date) &
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes')
            ]
        count = len(data)
        desc = f"Dormant accounts requiring statement suppression: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in freeze candidate detection: {e})"


def detect_transfer_candidates(df):
    """
    Detects dormant accounts inactive for 5 years requiring transfer to Central Bank
    per Article 8 of UAE regulation.
    """
    try:
        required_cols = ['Expected_Account_Dormant', 'Expected_Transfer_to_CB_Due']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Transfer Candidate check)"

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Expected_Transfer_to_CB_Due'].astype(str).str.lower() == 'yes')
            ]
        count = len(data)
        desc = f"Dormant accounts requiring transfer to Central Bank: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in transfer candidate detection: {e})"


def detect_foreign_currency_accounts(df):
    """
    Detects dormant accounts in foreign currencies that need conversion to AED
    before transfer to Central Bank per Article 8.5 of UAE regulation.
    """
    try:
        required_cols = ['Expected_Account_Dormant', 'Expected_Transfer_to_CB_Due', 'Currency']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Foreign Currency check)"

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Expected_Transfer_to_CB_Due'].astype(str).str.lower() == 'yes') &
            (~df['Currency'].astype(str).str.lower().isin(['aed', 'dirham', 'uae dirham']))
            ]
        count = len(data)
        desc = f"Foreign currency dormant accounts requiring AED conversion before CBUAE transfer: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in foreign currency detection: {e})"


def detect_safe_deposit_boxes(df):
    """
    Detects dormant safe deposit boxes with unpaid fees for over 3 years
    per Article 2.6 of UAE regulation.
    """
    try:
        required_cols = ['Account_Type', 'SDB_Charges_Outstanding', 'Date_SDB_Charges_Became_Outstanding',
                         'SDB_Tenant_Communication_Received']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Safe Deposit Box check)"

        # Convert date column to datetime
        if not pd.api.types.is_datetime64_dtype(df['Date_SDB_Charges_Became_Outstanding']):
            df['Date_SDB_Charges_Became_Outstanding'] = pd.to_datetime(df['Date_SDB_Charges_Became_Outstanding'],
                                                                       errors='coerce')

        three_years_ago = datetime.now() - timedelta(days=365 * 3)

        data = df[
            (df['Account_Type'].astype(str).str.lower() == 'safe_deposit_box') &
            (df['SDB_Charges_Outstanding'].astype(str).str.lower() == 'yes') &
            (df['Date_SDB_Charges_Became_Outstanding'] < three_years_ago) &
            (df['SDB_Tenant_Communication_Received'].astype(str).str.lower() != 'yes')
            ]
        count = len(data)
        desc = f"Safe deposit boxes with unpaid fees for over 3 years requiring court application: {count} boxes"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in safe deposit box detection: {e})"


def detect_unclaimed_payment_instruments(df):
    """
    Detects unclaimed bankers cheques, bank drafts, and cashier orders
    per Article 2 Second section of UAE regulation.
    """
    try:
        required_cols = ['Account_Type', 'Unclaimed_Item_Trigger_Date']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Unclaimed Payment Instruments check)"

        # Convert date column to datetime
        if not pd.api.types.is_datetime64_dtype(df['Unclaimed_Item_Trigger_Date']):
            df['Unclaimed_Item_Trigger_Date'] = pd.to_datetime(df['Unclaimed_Item_Trigger_Date'], errors='coerce')

        one_year_ago = datetime.now() - timedelta(days=365)

        data = df[
            (df['Account_Type'].astype(str).str.lower().isin(['bankers_cheque', 'bank_draft', 'cashier_order'])) &
            (df['Unclaimed_Item_Trigger_Date'] < one_year_ago)
            ]
        count = len(data)
        desc = f"Unclaimed payment instruments (1+ year since issuance): {count} items"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in unclaimed payment instruments detection: {e})"


def detect_claim_candidates(df):
    """
    Detects accounts with claims received that need to be processed
    per Article 4 of UAE regulation.
    """
    try:
        required_cols = ['Date_Claim_Received', 'Claim_Successful']
        if not all(col in df.columns for col in required_cols):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Claim Processing check)"

        # Convert date column to datetime
        if not pd.api.types.is_datetime64_dtype(df['Date_Claim_Received']):
            df['Date_Claim_Received'] = pd.to_datetime(df['Date_Claim_Received'], errors='coerce')

        # Get accounts with recent claims that haven't been processed
        data = df[
            (df['Date_Claim_Received'].notna()) &
            (df['Claim_Successful'].astype(str).str.lower().isin(['', 'pending', 'na', 'n/a', 'null', 'none']))
            ]
        count = len(data)
        desc = f"Customer claims requiring processing: {count} claims"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in claim candidates detection: {e})"


def log_flag_instructions(account_ids, agent_name, threshold_days):
    """
    Log flagging instructions to the dormant_flags table.
    """
    if not account_ids:
        return False, "No accounts to flag"

    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"

    try:
        with conn:
            cursor = conn.cursor()
            # Check if dormant_flags table exists
            cursor.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                return False, "Error: 'dormant_flags' table not found in the database"

            # Only insert if account_id does not already exist
            insert_sql = """
                         INSERT INTO dormant_flags(account_id, flag_instruction, timestamp)
                         SELECT ?, \
                                ?, \
                                ? WHERE NOT EXISTS (SELECT 1 FROM dormant_flags WHERE account_id = ?) \
                         """

            timestamp_now = datetime.now()
            rows_inserted = 0
            for acc_id in account_ids:
                # Ensure account_id is not None/empty string
                if pd.notna(acc_id) and str(acc_id).strip() != '':
                    cursor.execute(
                        insert_sql,
                        (
                            str(acc_id),
                            f"Identified by {agent_name} for review (Threshold: {threshold_days} days) - Regulation No. 1/2020",
                            timestamp_now,
                            str(acc_id)
                        )
                    )
                    rows_inserted += cursor.rowcount  # Count how many rows were actually inserted

            conn.commit()

            if rows_inserted < len(account_ids):
                return True, f"Logged {rows_inserted} unique accounts ({len(account_ids) - rows_inserted} were already in the flagging log)"
            else:
                return True, f"Logged {rows_inserted} unique accounts for flagging review!"

    except Exception as e:
        return False, f"DB logging failed: {e}"


def run_all_compliance_checks(df):
    """
    Run all compliance checks according to CBUAE regulation and return consolidated results.
    """
    # Define thresholds based on regulation
    three_years_ago = datetime.now() - timedelta(days=365 * 3)

    results = {
        "total_accounts": len(df),
        "contact": {"df": None, "count": 0, "desc": ""},
        "flag": {"df": None, "count": 0, "desc": ""},
        "ledger": {"df": None, "count": 0, "desc": ""},
        "freeze": {"df": None, "count": 0, "desc": ""},
        "transfer": {"df": None, "count": 0, "desc": ""},
        "foreign_currency": {"df": None, "count": 0, "desc": ""},
        "safe_deposit": {"df": None, "count": 0, "desc": ""},
        "payment_instruments": {"df": None, "count": 0, "desc": ""},
        "claims": {"df": None, "count": 0, "desc": ""}
    }

    # Run each check
    results["contact"]["df"], results["contact"]["count"], results["contact"]["desc"] = detect_incomplete_contact(df)
    results["flag"]["df"], results["flag"]["count"], results["flag"]["desc"] = detect_flag_candidates(df,
                                                                                                      three_years_ago)
    results["ledger"]["df"], results["ledger"]["count"], results["ledger"]["desc"] = detect_ledger_candidates(df)
    results["freeze"]["df"], results["freeze"]["count"], results["freeze"]["desc"] = detect_freeze_candidates(df,
                                                                                                              three_years_ago)
    results["transfer"]["df"], results["transfer"]["count"], results["transfer"]["desc"] = detect_transfer_candidates(
        df)
    results["foreign_currency"]["df"], results["foreign_currency"]["count"], results["foreign_currency"][
        "desc"] = detect_foreign_currency_accounts(df)
    results["safe_deposit"]["df"], results["safe_deposit"]["count"], results["safe_deposit"][
        "desc"] = detect_safe_deposit_boxes(df)
    results["payment_instruments"]["df"], results["payment_instruments"]["count"], results["payment_instruments"][
        "desc"] = detect_unclaimed_payment_instruments(df)
    results["claims"]["df"], results["claims"]["count"], results["claims"]["desc"] = detect_claim_candidates(df)

    return results