import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from database.connection import get_db_connection


def detect_incomplete_contact(df):
    """Detects accounts that require contact attempts per CBUAE Article 3."""
    try:
        # Check required columns for this check
        required_cols = ['Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Expected_Account_Dormant']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Find accounts marked as potentially dormant but no contact attempt made
        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower() == 'no')
            ]

        count = len(data)
        desc = f"Accounts requiring contact attempts (CBUAE Article 3): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in contact attempt verification: {e})"


def detect_flag_candidates(df, threshold_date):
    """Detects accounts meeting dormancy criteria not yet flagged (CBUAE Article 2)."""
    try:
        # Check required columns
        required_cols = ['Date_Last_Cust_Initiated_Activity', 'Expected_Account_Dormant']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Find accounts inactive before threshold but not yet flagged as dormant
        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date) &
            (df['Expected_Account_Dormant'].astype(str).str.lower() != 'yes')
            ]

        count = len(data)
        desc = f"Accounts inactive over threshold, not yet flagged dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in dormant flag detection: {e})"


def detect_ledger_candidates(df):
    """Detects accounts requiring ledger classification (CBUAE Article 3)."""
    try:
        # Check required columns
        if 'Expected_Account_Dormant' not in df.columns:
            return pd.DataFrame(), 0, "(Skipped: Required column missing for Ledger Candidate check)"

        # Find accounts flagged as dormant that need ledger classification
        data = df[df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes'].copy()

        # Check if these accounts are already in the dormant_ledger table
        ids_in_ledger = []
        conn = get_db_connection()
        if conn:
            try:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT account_id FROM dormant_ledger")
                    ids_in_ledger = [row[0] for row in cursor.fetchall()]
            except Exception as db_e:
                st.warning(f"Could not check dormant ledger table: {db_e}. Proceeding without filtering.")
                ids_in_ledger = []

        # Filter out accounts already in the ledger
        if ids_in_ledger:
            data = data[~data['Account_ID'].isin(ids_in_ledger)]

        count = len(data)
        desc = f"Dormant accounts requiring ledger classification: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in ledger candidate detection: {e})"


def detect_freeze_candidates(df, threshold_date):
    """Detects dormant accounts inactive beyond freeze threshold (CBUAE Article 3.5)."""
    try:
        # Check required columns
        required_cols = ['Date_Last_Cust_Initiated_Activity', 'Expected_Account_Dormant',
                         'Expected_Requires_Article_3_Process']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Find dormant accounts that have been dormant for longer than threshold
        # and require Article 3 process but haven't been processed
        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_date) &
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Expected_Requires_Article_3_Process'].astype(str).str.lower() == 'yes')
            ]

        count = len(data)
        desc = f"Dormant accounts requiring Article 3 freeze process: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in freeze candidate detection: {e})"


def detect_transfer_candidates(df, cutoff_date):
    """
    Detects dormant accounts that should be transferred to Central Bank (CBUAE Article 8).
    Accounts dormant for 5+ years
    """
    if not isinstance(cutoff_date, datetime):
        return pd.DataFrame(), 0, "(Skipped: Valid cutoff date not provided for Transfer check)"

    try:
        # Check required columns
        required_cols = ['Date_Last_Cust_Initiated_Activity', 'Expected_Account_Dormant', 'Expected_Transfer_to_CB_Due']

        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped: Missing columns: {', '.join(missing)})"

        # Find accounts inactive before cutoff that should be transferred to CB
        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < cutoff_date) &
            (df['Expected_Account_Dormant'].astype(str).str.lower() == 'yes') &
            (df['Expected_Transfer_to_CB_Due'].astype(str).str.lower() != 'completed')
            ]

        count = len(data)
        desc = f"Dormant accounts due for transfer to Central Bank (5+ years): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in transfer candidate detection: {e})"


def log_flag_instructions(account_ids, agent_name, threshold_days):
    """
    Log flagging instructions to the dormant_flags table.

    Args:
        account_ids (list): List of account IDs to flag
        agent_name (str): Name of the agent that detected these accounts
        threshold_days (int): The threshold in days used for detection

    Returns:
        tuple: (success, message) with success being a boolean and message a status string
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
                            f"Identified by {agent_name} as dormant per CBUAE criteria (Threshold: {threshold_days} days)",
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


def run_all_compliance_checks(df, general_threshold_date, freeze_threshold_date, cbuae_cutoff_date):
    """
    Run all compliance checks and return a consolidated result.

    Args:
        df (pandas.DataFrame): The account data
        general_threshold_date (datetime): Date threshold for flagging dormant accounts
        freeze_threshold_date (datetime): Date threshold for freezing accounts
        cbuae_cutoff_date (datetime): Cutoff date for CBUAE transfers

    Returns:
        dict: Dictionary containing results from all compliance checks
    """
    results = {
        "total_accounts": len(df),
        "contact": {"df": None, "count": 0, "desc": ""},
        "flag": {"df": None, "count": 0, "desc": ""},
        "ledger": {"df": None, "count": 0, "desc": ""},
        "freeze": {"df": None, "count": 0, "desc": ""},
        "transfer": {"df": None, "count": 0, "desc": ""}
    }

    # Run each check
    results["contact"]["df"], results["contact"]["count"], results["contact"]["desc"] = detect_incomplete_contact(df)
    results["flag"]["df"], results["flag"]["count"], results["flag"]["desc"] = detect_flag_candidates(df,
                                                                                                      general_threshold_date)
    results["ledger"]["df"], results["ledger"]["count"], results["ledger"]["desc"] = detect_ledger_candidates(df)
    results["freeze"]["df"], results["freeze"]["count"], results["freeze"]["desc"] = detect_freeze_candidates(df,
                                                                                                              freeze_threshold_date)

    # Only run transfer check if valid cutoff date is provided
    if isinstance(cbuae_cutoff_date, datetime):
        results["transfer"]["df"], results["transfer"]["count"], results["transfer"][
            "desc"] = detect_transfer_candidates(df, cbuae_cutoff_date)
    else:
        results["transfer"]["df"], results["transfer"]["count"], results["transfer"]["desc"] = (pd.DataFrame(), 0,
                                                                                                "Skipped: Invalid CBUAE cutoff date")

    return results