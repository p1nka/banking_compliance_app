import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from database.connection import get_db_connection


def detect_incomplete_contact(df):
    """Detects accounts with at least one 'No' contact attempt."""
    try:
        if not all(col in df.columns for col in ['Email_Contact_Attempt', 'SMS_Contact_Attempt', 'Phone_Call_Attempt']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Incomplete Contact check)"

        data = df[
            (df['Email_Contact_Attempt'].astype(str).str.lower() == 'no') |
            (df['SMS_Contact_Attempt'].astype(str).str.lower() == 'no') |
            (df['Phone_Call_Attempt'].astype(str).str.lower() == 'no')
            ]
        count = len(data)
        desc = f"Accounts with incomplete contact attempts: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in contact attempt verification: {e})"


def detect_flag_candidates(df, threshold_date):
    """Detects accounts inactive over threshold, not yet flagged dormant."""
    try:
        if not all(col in df.columns for col in ['Last_Transaction_Date', 'Account_Status']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Flag Candidate check)"

        data = df[
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Account_Status'].astype(str).str.lower() != 'dormant')
            ]
        count = len(data)
        desc = f"Accounts inactive over threshold, not yet flagged dormant: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in dormant flag detection: {e})"


def detect_ledger_candidates(df):
    """Detects accounts marked dormant requiring ledger classification (not already in ledger)."""
    try:
        if 'Account_Status' not in df.columns:
            return pd.DataFrame(), 0, "(Skipped: Required column missing for Ledger Candidate check)"

        data = df[df['Account_Status'].astype(
            str).str.lower() == 'dormant'].copy()  # Work on copy to avoid SettingWithCopyWarning

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
        desc = f"Dormant accounts requiring ledger classification (not yet in ledger): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in ledger candidate detection: {e})"


def detect_freeze_candidates(df, threshold_date):
    """Detects dormant accounts inactive beyond freeze threshold."""
    try:
        if not all(col in df.columns for col in ['Last_Transaction_Date', 'Account_Status']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Freeze Candidate check)"

        data = df[
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < threshold_date) &
            (df['Account_Status'].astype(str).str.lower() == 'dormant')
            ]
        count = len(data)
        desc = f"Dormant accounts inactive beyond freeze threshold: {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in freeze candidate detection: {e})"


def detect_transfer_candidates(df, cutoff_date):
    """Detects dormant accounts inactive before a specific cutoff date (e.g., CBUAE)."""
    if not isinstance(cutoff_date, datetime):
        return pd.DataFrame(), 0, "(Skipped: Valid cutoff date not provided for Transfer check)"
    try:
        if not all(col in df.columns for col in ['Last_Transaction_Date', 'Account_Status']):
            return pd.DataFrame(), 0, "(Skipped: Required columns missing for Transfer Candidate check)"

        data = df[
            (df['Last_Transaction_Date'].notna()) &
            (df['Last_Transaction_Date'] < cutoff_date) &
            (df['Account_Status'].astype(str).str.lower() == 'dormant')
            ]
        count = len(data)
        desc = f"Dormant accounts inactive before cutoff ({cutoff_date.strftime('%Y-%m-%d')}): {count} accounts"
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
                            f"Identified by {agent_name} for review (Threshold: {threshold_days} days)",
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