import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
# Assuming get_db_connection is correctly set up in your project
from database.connection import get_db_connection
import pyodbc  # Added import for pyodbc

# CBUAE Compliance Periods/Values (can be in config.py)
THREE_MONTH_WAIT_DAYS = 90
RECORD_RETENTION_YEARS_POLICY = 7  # Example bank policy, CBUAE is often perpetual post-transfer


def detect_incomplete_contact_attempts(df):
    """
    Detects accounts where contact attempts are insufficient as per CBUAE Art. 3.1 / Art. 5.
    This check focuses on accounts expected to be dormant or needing contact.
    Assumes columns like 'Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Customer_Responded_to_Bank_Contact'
    are present from the schema or set by data standardization.
    Also considers 'Expected_Account_Dormant' which might be set by dormant agents.
    """
    try:
        required_cols = [
            'Account_ID', 'Expected_Account_Dormant',  # From dormant agent / initial data
            'Bank_Contact_Attempted_Post_Dormancy_Trigger',  # From schema
            'Customer_Responded_to_Bank_Contact'  # From schema
        ]
        # Your CSV uses Email_Contact_Attempt, SMS_Contact_Attempt, Phone_Call_Attempt
        # Let's adapt to use those if 'Bank_Contact_Attempted_Post_Dormancy_Trigger' is not the primary indicator
        # CBUAE requires *multiple* channels. This checks if *any* required contact hasn't happened or failed.

        contact_method_cols = ['Email_Contact_Attempt', 'SMS_Contact_Attempt',
                               'Phone_Call_Attempt']  # From your schema.py

        # Check if core required columns exist
        if not ('Account_ID' in df.columns and 'Expected_Account_Dormant' in df.columns and \
                all(col in df.columns for col in contact_method_cols)):
            missing = [c for c in ['Account_ID', 'Expected_Account_Dormant'] + contact_method_cols if
                       c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Incomplete Contact: Missing {', '.join(missing)})"

        # Condition: Account is expected to be dormant OR bank has already started a dormancy-related contact
        # AND at least one of the primary contact methods shows 'no' (or implies failure)
        # This is a simplification. True CBUAE compliance would check *sufficiency* of attempts.
        # The provided `detect_incomplete_contact` looks for at least one 'no'.

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (
                    (df['Email_Contact_Attempt'].astype(str).str.lower().isin(['no', 'nan', ''])) |
                    (df['SMS_Contact_Attempt'].astype(str).str.lower().isin(['no', 'nan', ''])) |
                    (df['Phone_Call_Attempt'].astype(str).str.lower().isin(['no', 'nan', '']))
                # Add 'Customer_Responded_to_Bank_Contact' == 'no' if bank *did* attempt all
                # This gets complex quickly and depends on how Bank_Contact_Attempted_Post_Dormancy_Trigger is used.
            )
            ].copy()

        count = len(data)
        desc = f"Accounts (expected dormant) with potentially incomplete contact attempts (CBUAE Art 3.1/5): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Incomplete Contact check: {e})"


def detect_unflagged_dormant_candidates(df, inactivity_threshold_date):
    """
    Detects accounts inactive over CBUAE threshold (Art. 2), not yet flagged as 'Expected_Account_Dormant'.
    This is a crucial compliance check.
    Uses: Date_Last_Cust_Initiated_Activity, Account_Type, Unclaimed_Item_Trigger_Date (for instruments)
          and checks against 'Expected_Account_Dormant'.
    """
    try:
        required_cols_general = ['Account_ID', 'Date_Last_Cust_Initiated_Activity', 'Account_Type',
                                 'Expected_Account_Dormant']
        required_cols_instr = ['Unclaimed_Item_Trigger_Date']  # For payment instruments

        if not all(col in df.columns for col in required_cols_general):
            missing = [c for c in required_cols_general if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Unflagged Candidates: Missing general cols {', '.join(missing)})"

        # Convert date columns
        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'],
                                                                     errors='coerce')
        if 'Unclaimed_Item_Trigger_Date' in df.columns and not pd.api.types.is_datetime64_dtype(
                df['Unclaimed_Item_Trigger_Date']):
            df['Unclaimed_Item_Trigger_Date'] = pd.to_datetime(df['Unclaimed_Item_Trigger_Date'], errors='coerce')

        # Standard accounts (e.g., 3 years)
        standard_accounts_unflagged = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < inactivity_threshold_date) &
            (~df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (~df['Account_Type'].astype(str).str.lower().isin(
                ['bankers_cheque', 'bank_draft', 'cashier_order', 'safe_deposit_box']))
            # Exclude types handled separately for dormancy trigger
            ].copy()

        # Payment instruments (e.g., 1 year from Unclaimed_Item_Trigger_Date)
        payment_instruments_unflagged_list = []
        if all(col in df.columns for col in required_cols_instr):
            one_year_ago_instr = datetime.now() - timedelta(days=365 * 1)  # CBUAE Art 2.4
            payment_instruments_unflagged = df[
                (df['Account_Type'].astype(str).str.lower().isin(['bankers_cheque', 'bank_draft', 'cashier_order'])) &
                (df['Unclaimed_Item_Trigger_Date'].notna()) &
                (df['Unclaimed_Item_Trigger_Date'] < one_year_ago_instr) &
                (~df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1']))
                ].copy()
            if not payment_instruments_unflagged.empty:
                payment_instruments_unflagged_list.append(payment_instruments_unflagged)

        all_unflagged = pd.concat([standard_accounts_unflagged] + payment_instruments_unflagged_list).drop_duplicates(
            subset=['Account_ID'])
        count = len(all_unflagged)
        desc = f"Accounts meeting dormancy criteria but NOT FLAGGED 'Expected_Account_Dormant' (CBUAE Art. 2): {count} accounts"
        return all_unflagged, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unflagged Dormant Candidates check: {e})"


def detect_internal_ledger_candidates(df):
    """
    Detects accounts flagged 'Expected_Account_Dormant' AND 'Expected_Requires_Article_3_Process',
    AND 3-month waiting period after last bank contact is over,
    that are NOT YET in the internal 'dormant_ledger' table (CBUAE Art. 3.4, 3.5).
    Uses: Account_ID, Expected_Account_Dormant, Expected_Requires_Article_3_Process, Date_Last_Bank_Contact_Attempt
    """
    try:
        required_cols = [
            'Account_ID', 'Expected_Account_Dormant',
            'Expected_Requires_Article_3_Process', 'Date_Last_Bank_Contact_Attempt'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Internal Ledger: Missing {', '.join(missing)})"

        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Bank_Contact_Attempt']):
            df['Date_Last_Bank_Contact_Attempt'] = pd.to_datetime(df['Date_Last_Bank_Contact_Attempt'], errors='coerce')

        three_months_ago = datetime.now() - timedelta(days=THREE_MONTH_WAIT_DAYS)

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['Expected_Requires_Article_3_Process'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['Date_Last_Bank_Contact_Attempt'].notna()) &
            (df['Date_Last_Bank_Contact_Attempt'] < three_months_ago)  # Waiting period passed
            ].copy()

        ids_in_ledger = []
        conn = get_db_connection()
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT account_id FROM dormant_ledger")
                    ids_in_ledger = [row[0] for row in cursor.fetchall()]
            except Exception as db_e:
                # FIX: A failed SELECT does not require a rollback.
                # Simply warn the user and proceed. The faulty rollback logic is removed.
                st.sidebar.warning(f"Could not check dormant_ledger table: {db_e}. Proceeding without filtering.")

        if ids_in_ledger:
            data = data[~data['Account_ID'].isin(ids_in_ledger)]

        count = len(data)
        desc = f"Dormant accounts for INTERNAL LEDGER transfer (Art. 3.5, post {THREE_MONTH_WAIT_DAYS}-day wait): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Internal Ledger Candidates check: {e})"


def detect_statement_freeze_candidates(df, inactivity_threshold_date_for_freeze):
    """
    Detects accounts flagged 'Expected_Account_Dormant' that should have statement generation suppressed (CBUAE Art. 7.3).
    Uses: Expected_Account_Dormant, Date_Last_Cust_Initiated_Activity
          (And a schema column like 'Statement_Suppression_Active' if available to check current status)
    """
    try:
        required_cols = ['Account_ID', 'Expected_Account_Dormant', 'Date_Last_Cust_Initiated_Activity']
        # Optional: 'Statement_Suppression_Active' to check if already done
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Statement Freeze: Missing {', '.join(missing)})"

        if not pd.api.types.is_datetime64_dtype(df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'],
                                                                     errors='coerce')

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            # Typically freeze applies once dormant, sometimes after a deeper inactivity period
            (df['Date_Last_Cust_Initiated_Activity'] < inactivity_threshold_date_for_freeze)
            # Optionally, add: & (df.get('Statement_Suppression_Active', pd.Series(dtype=str)).astype(str).str.lower() != 'yes')
            ].copy()
        count = len(data)
        desc = f"Dormant accounts requiring statement suppression (Art. 7.3): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Statement Freeze Candidates check: {e})"


def detect_cbuae_transfer_candidates(df):
    """
    Detects accounts/balances flagged 'Expected_Transfer_to_CB_Due' (CBUAE Art. 8).
    This relies on dormant agents or a master process setting this flag correctly
    based on 5-year inactivity and other Art. 8.1 conditions.
    Uses: Account_ID, Expected_Account_Dormant, Expected_Transfer_to_CB_Due
    """
    try:
        required_cols = ['Account_ID', 'Expected_Account_Dormant', 'Expected_Transfer_to_CB_Due']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped CBUAE Transfer: Missing {', '.join(missing)})"

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['Expected_Transfer_to_CB_Due'].astype(str).str.lower().isin(['yes', 'true', '1']))
            ].copy()
        count = len(data)
        desc = f"Dormant accounts/balances due for CBUAE transfer (Art. 8): {count} items"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in CBUAE Transfer Candidates check: {e})"


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
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                return False, "Error: 'dormant_flags' table not found in the database. Run schema initialization."

            insert_sql = """
                         INSERT INTO dormant_flags (account_id, flag_instruction, timestamp)
                         VALUES (%s, %s, %s)
                         """

            timestamp_now = datetime.now()
            rows_inserted = 0
            skipped_due_to_existence = 0

            for acc_id in account_ids:
                if pd.notna(acc_id) and str(acc_id).strip() != '':
                    try:
                        cursor.execute("SELECT 1 FROM dormant_flags WHERE account_id = %s", (str(acc_id),))
                        if cursor.fetchone():
                            skipped_due_to_existence += 1
                            continue

                        cursor.execute(
                            insert_sql,
                            (
                                str(acc_id),
                                f"Identified by {agent_name} for review (Threshold: {threshold_days} days) - CBUAE",
                                timestamp_now
                            )
                        )
                        rows_inserted += cursor.rowcount

                    except Exception as e_row:
                        if "PRIMARY KEY constraint" in str(e_row):
                            skipped_due_to_existence += 1
                        else:
                            st.sidebar.warning(f"Skipping log for {acc_id} due to DB error: {e_row}")

            # FIX: Only commit if rows were actually inserted. This prevents the
            # "Cannot commit transaction" error if no new data was added.
            if rows_inserted > 0:
                conn.commit()

            if skipped_due_to_existence > 0:
                return True, f"Logged {rows_inserted} new unique accounts. {skipped_due_to_existence} were already in the flagging log or caused an error."
            else:
                return True, f"Logged {rows_inserted} unique accounts for flagging review!"

    except Exception as e:
        # The original error happens before this block is reached. If an error occurs
        # during the process (e.g., connection lost), a rollback is still a good idea.
        # But we must handle the case where the connection is already closed.
        if conn:
            try:
                conn.rollback()
            except Exception as rollback_e:
                # Avoid showing a secondary error if the rollback itself fails
                st.sidebar.error(f"DB logging failed: {e}. Additionally, rollback failed: {rollback_e}")
                return False, f"DB logging failed: {e}"

        return False, f"DB logging failed: {e}"


# Additional compliance agents
def detect_flag_candidates(df, inactivity_threshold_date):
    """Alias for detect_unflagged_dormant_candidates for UI compatibility"""
    return detect_unflagged_dormant_candidates(df, inactivity_threshold_date)


def detect_ledger_candidates(df):
    """Alias for detect_internal_ledger_candidates for UI compatibility"""
    return detect_internal_ledger_candidates(df)


def detect_freeze_candidates(df, inactivity_threshold_date_for_freeze):
    """Alias for detect_statement_freeze_candidates for UI compatibility"""
    return detect_statement_freeze_candidates(df, inactivity_threshold_date_for_freeze)


def detect_transfer_candidates_to_cb(df):
    """Alias for detect_cbuae_transfer_candidates for UI compatibility"""
    return detect_cbuae_transfer_candidates(df)


def detect_foreign_currency_conversion_needed(df):
    """
    Detects dormant accounts in foreign currency that need conversion to AED for CBUAE transfer (Art. 8.5).
    """
    try:
        required_cols = ['Account_ID', 'Expected_Account_Dormant', 'Expected_Transfer_to_CB_Due', 'Currency',
                         'Current_Balance']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped FX Conversion: Missing {', '.join(missing)})"

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['Expected_Transfer_to_CB_Due'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (~df['Currency'].astype(str).str.upper().isin(['AED', 'DIRHAM', 'DIRHAMS', ''])) &
            (pd.to_numeric(df['Current_Balance'], errors='coerce').fillna(0) > 0)
            ].copy()
        count = len(data)
        desc = f"Foreign currency dormant accounts needing AED conversion for CBUAE transfer (Art. 8.5): {count} accounts"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Foreign Currency Conversion check: {e})"


def detect_sdb_court_application_needed(df):
    """
    Detects Safe Deposit Boxes requiring court application (Art. 3.7).
    """
    try:
        required_cols = [
            'Account_ID', 'Account_Type', 'Expected_Account_Dormant',
            'SDB_Charges_Outstanding', 'SDB_Court_Application_Submitted'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped SDB Court Application: Missing {', '.join(missing)})"

        data = df[
            (df['Account_Type'].astype(str).str.contains("Safe Deposit", case=False, na=False)) &
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (pd.to_numeric(df['SDB_Charges_Outstanding'], errors='coerce').fillna(0) > 0) &
            (~df['SDB_Court_Application_Submitted'].astype(str).str.lower().isin(['yes', 'true', '1']))
            ].copy()
        count = len(data)
        desc = f"Safe Deposit Boxes requiring court application (Art. 3.7): {count} boxes"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in SDB Court Application check: {e})"


def detect_unclaimed_payment_instruments_ledger(df):
    """
    Detects unclaimed payment instruments for internal ledger (Art. 3.6).
    """
    try:
        required_cols = [
            'Account_ID', 'Account_Type', 'Expected_Account_Dormant',
            'Unclaimed_Item_Trigger_Date', 'Moved_To_Internal_Dormant_Ledger'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Unclaimed Instruments Ledger: Missing {', '.join(missing)})"

        if not pd.api.types.is_datetime64_dtype(df['Unclaimed_Item_Trigger_Date']):
            df['Unclaimed_Item_Trigger_Date'] = pd.to_datetime(df['Unclaimed_Item_Trigger_Date'], errors='coerce')

        data = df[
            (df['Account_Type'].astype(str).str.contains("Cheque|Draft|Order", case=False, na=False)) &
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['Unclaimed_Item_Trigger_Date'].notna()) &
            (~df['Moved_To_Internal_Dormant_Ledger'].astype(str).str.lower().isin(['yes', 'true', '1']))
            ].copy()
        count = len(data)
        desc = f"Unclaimed payment instruments for internal ledger (Art. 3.6): {count} instruments"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unclaimed Instruments Ledger check: {e})"


def detect_claim_processing_pending(df):
    """
    Detects customer claims (>1 month old) pending processing (Art. 4).
    """
    try:
        required_cols = [
            'Account_ID', 'Customer_Claim_Date', 'Claim_Status'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Claims Processing: Missing {', '.join(missing)})"

        if not pd.api.types.is_datetime64_dtype(df['Customer_Claim_Date']):
            df['Customer_Claim_Date'] = pd.to_datetime(df['Customer_Claim_Date'], errors='coerce')

        one_month_ago = datetime.now() - timedelta(days=30)

        data = df[
            (df['Customer_Claim_Date'].notna()) &
            (df['Customer_Claim_Date'] < one_month_ago) &
            (~df['Claim_Status'].astype(str).str.lower().isin(['completed', 'processed', 'resolved', 'rejected']))
            ].copy()
        count = len(data)
        desc = f"Customer claims (>1 month old) pending processing (Art. 4): {count} claims"
        return data, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Claims Processing check: {e})"


def generate_annual_cbuae_report_summary(df):
    """
    Generates summary for CBUAE Annual Report (Art. 3.10).
    """
    try:
        required_cols = [
            'Account_ID', 'Expected_Account_Dormant', 'Current_Balance', 'Currency',
            'Expected_Transfer_to_CB_Due', 'Transferred_to_CBUAE_Date'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Annual Report: Missing {', '.join(missing)})"

        # Get accounts flagged dormant
        dormant_accounts = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1']))
        ].copy()

        # Get accounts flagged for CB transfer but not yet transferred
        pending_cb_transfer = df[
            (df['Expected_Transfer_to_CB_Due'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (~df['Transferred_to_CBUAE_Date'].notna())
            ].copy()

        # Get accounts already transferred to CB
        transferred_to_cb = df[
            (df['Transferred_to_CBUAE_Date'].notna())
        ].copy()

        # Create report summary data
        this_year = datetime.now().year
        summary_data = {
            'ReportYear': this_year,
            'TotalDormantAccounts': len(dormant_accounts),
            'TotalDormantBalance': dormant_accounts['Current_Balance'].astype(float).sum(),
            'PendingCBTransfer': len(pending_cb_transfer),
            'PendingCBTransferBalance': pending_cb_transfer['Current_Balance'].astype(float).sum(),
            'TransferredToCB': len(transferred_to_cb),
            'TransferredToCBBalance': transferred_to_cb['Current_Balance'].astype(float).sum(),
            'ReportGenerationDate': datetime.now()
        }

        # Convert to DataFrame for the report
        summary_df = pd.DataFrame([summary_data])

        count = 1  # Just one summary row
        desc = f"Annual CBUAE Report Summary for {this_year} generated (Art. 3.10): {len(dormant_accounts)} dormant accounts"
        return summary_df, count, desc
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Annual Report Summary check: {e})"


def check_record_retention_compliance(df):
    """
    Checks record retention compliance (Art. 3.9 related).
    Returns two DataFrames:
    1. Records not meeting policy retention requirement (not compliant)
    2. Compliant records (meets retention or CBUAE perpetual)
    """
    try:
        required_cols = [
            'Account_ID', 'Expected_Transfer_to_CB_Due', 'Transferred_to_CBUAE_Date',
            'Date_Last_Cust_Initiated_Activity', 'Record_Retention_End_Date'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            return pd.DataFrame(), pd.DataFrame(), f"(Skipped Record Retention: Missing {', '.join(missing)})"

        # Convert date columns
        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Record_Retention_End_Date', 'Transferred_to_CBUAE_Date']
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Calculate records that might need retention
        # 1. Items transferred to CBUAE - perpetual retention
        cbuae_transferred = df[df['Transferred_to_CBUAE_Date'].notna()].copy()

        # 2. Items not transferred to CBUAE - retention based on bank policy
        bank_policy_records = df[
            (~df['Transferred_to_CBUAE_Date'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'].notna())
            ].copy()

        # For bank policy records, calculate retention end date if not already populated
        # Default policy is to keep records for RECORD_RETENTION_YEARS_POLICY years after last activity
        if not bank_policy_records.empty:
            bank_policy_records.loc[:, 'Calculated_Retention_End'] = bank_policy_records[
                                                                         'Date_Last_Cust_Initiated_Activity'] + \
                                                                     pd.Timedelta(
                                                                         days=365 * RECORD_RETENTION_YEARS_POLICY)

            # If Record_Retention_End_Date is populated, use that instead
            bank_policy_records.loc[:, 'Effective_Retention_End'] = bank_policy_records.apply(
                lambda x: x['Record_Retention_End_Date'] if pd.notna(x['Record_Retention_End_Date'])
                else x['Calculated_Retention_End'], axis=1
            )

            # Records not compliant with policy - retention date passed
            not_compliant_policy = bank_policy_records[
                bank_policy_records['Effective_Retention_End'] < datetime.now()
                ].copy()

            # Records compliant with policy - retention date in future
            compliant_policy = bank_policy_records[
                bank_policy_records['Effective_Retention_End'] >= datetime.now()
                ].copy()
        else:
            not_compliant_policy = pd.DataFrame()
            compliant_policy = pd.DataFrame()

        # Combine compliant: Policy-compliant records and CBUAE transferred (perpetual)
        compliant_df = pd.concat([compliant_policy, cbuae_transferred]).drop_duplicates(subset=['Account_ID'])

        count = len(not_compliant_policy)
        desc = f"Records potentially not compliant with retention policy (Art. 3.9): {count} records"
        return not_compliant_policy, compliant_df, desc
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), f"(Error in Record Retention Compliance check: {e})"


def run_all_compliance_checks(df, general_threshold_date, freeze_threshold_date, cbuae_cutoff_date_ignored=None,
                              agent_name="ComplianceSystem"):
    """
    Run all compliance checks.
    'cbuae_cutoff_date' is not directly used here as 'detect_cbuae_transfer_candidates' relies on a pre-set flag.
    'general_threshold_date' is for 'detect_unflagged_dormant_candidates'.
    'freeze_threshold_date' is for 'detect_statement_freeze_candidates'.
    """
    results = {
        "total_accounts_processed": len(df),
        "incomplete_contact": {},
        "flag_candidates": {},
        "ledger_candidates_internal": {},
        "statement_freeze_needed": {},
        "transfer_candidates_cb": {},
        "fx_conversion_needed": {},
        "sdb_court_application": {},
        "unclaimed_instruments_ledger": {},
        "claims_processing_pending": {},
        "annual_cbuae_report": {},
        "record_retention_check": {},
        "flag_logging_status": {}
    }
    df_copy = df.copy()  # Work on a copy to avoid modifying original df

    # Run all the compliance checks
    results["incomplete_contact"]["df"], results["incomplete_contact"]["count"], results["incomplete_contact"]["desc"] = \
        detect_incomplete_contact_attempts(df_copy)

    results["flag_candidates"]["df"], results["flag_candidates"]["count"], results["flag_candidates"]["desc"] = \
        detect_flag_candidates(df_copy, general_threshold_date)

    # Log if unflagged candidates found
    if results["flag_candidates"]["count"] > 0 and 'Account_ID' in results["flag_candidates"]["df"].columns:
        ids_to_log = results["flag_candidates"]["df"]['Account_ID'].tolist()
        # Determine threshold_days based on general_threshold_date for logging
        # This is an approximation for the log; individual records might have different effective thresholds (e.g. instruments)
        log_threshold_days = (datetime.now() - general_threshold_date).days
        status, msg = log_flag_instructions(ids_to_log, agent_name, log_threshold_days)
        results["flag_logging_status"] = {"status": status, "message": msg}
    else:
        results["flag_logging_status"] = {"status": True,
                                          "message": "No unflagged candidates to log or Account_ID missing."}

    results["ledger_candidates_internal"]["df"], results["ledger_candidates_internal"]["count"], \
        results["ledger_candidates_internal"]["desc"] = \
        detect_ledger_candidates(df_copy)

    results["statement_freeze_needed"]["df"], results["statement_freeze_needed"]["count"], \
        results["statement_freeze_needed"]["desc"] = \
        detect_freeze_candidates(df_copy, freeze_threshold_date)

    results["transfer_candidates_cb"]["df"], results["transfer_candidates_cb"]["count"], \
        results["transfer_candidates_cb"]["desc"] = \
        detect_transfer_candidates_to_cb(df_copy)  # Relies on 'Expected_Transfer_to_CB_Due'

    results["fx_conversion_needed"]["df"], results["fx_conversion_needed"]["count"], results["fx_conversion_needed"][
        "desc"] = \
        detect_foreign_currency_conversion_needed(df_copy)

    results["sdb_court_application"]["df"], results["sdb_court_application"]["count"], results["sdb_court_application"][
        "desc"] = \
        detect_sdb_court_application_needed(df_copy)

    results["unclaimed_instruments_ledger"]["df"], results["unclaimed_instruments_ledger"]["count"], \
        results["unclaimed_instruments_ledger"]["desc"] = \
        detect_unclaimed_payment_instruments_ledger(df_copy)

    results["claims_processing_pending"]["df"], results["claims_processing_pending"]["count"], \
        results["claims_processing_pending"]["desc"] = \
        detect_claim_processing_pending(df_copy)

    results["annual_cbuae_report"]["df"], results["annual_cbuae_report"]["count"], results["annual_cbuae_report"][
        "desc"] = \
        generate_annual_cbuae_report_summary(df_copy)

    # Special handling for record retention which returns two dataframes
    not_compliant_df, compliant_df, retention_desc = check_record_retention_compliance(df_copy)
    results["record_retention_check"] = {
        "df": not_compliant_df,
        "compliant_df": compliant_df,
        "count": len(not_compliant_df),
        "desc": retention_desc
    }

    return results