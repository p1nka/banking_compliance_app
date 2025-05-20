# --- START OF REFINED agents/compliance.py ---
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
# Assuming get_db_connection is correctly set up in your project
from database.connection import get_db_connection  # Or your specific path

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
                with conn.cursor() as cursor:  # Use 'with' for cursor too
                    cursor.execute("SELECT account_id FROM dormant_ledger")
                    ids_in_ledger = [row[0] for row in cursor.fetchall()]
            except Exception as db_e:
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


# (log_flag_instructions remains largely the same as your provided version, ensure it takes account_ids (list), agent_name, threshold_days)
# The log_flag_instructions in your provided compliance.py looks fine for its purpose.

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
        "incomplete_contact": {}, "unflagged_dormant": {},
        "internal_ledger": {}, "statement_freeze": {}, "cbuae_transfer": {},
        "flag_logging_status": {}
    }
    df_copy = df.copy()

    results["incomplete_contact"]["df"], results["incomplete_contact"]["count"], results["incomplete_contact"]["desc"] = \
        detect_incomplete_contact_attempts(df_copy)

    results["unflagged_dormant"]["df"], results["unflagged_dormant"]["count"], results["unflagged_dormant"]["desc"] = \
        detect_unflagged_dormant_candidates(df_copy, general_threshold_date)

    # Log if unflagged candidates found
    if results["unflagged_dormant"]["count"] > 0 and 'Account_ID' in results["unflagged_dormant"]["df"].columns:
        ids_to_log = results["unflagged_dormant"]["df"]['Account_ID'].tolist()
        # Determine threshold_days based on general_threshold_date for logging
        # This is an approximation for the log; individual records might have different effective thresholds (e.g. instruments)
        log_threshold_days = (datetime.now() - general_threshold_date).days
        status, msg = log_flag_instructions(ids_to_log, agent_name, log_threshold_days)
        results["flag_logging_status"] = {"status": status, "message": msg}
    else:
        results["flag_logging_status"] = {"status": True,
                                          "message": "No unflagged candidates to log or Account_ID missing."}

    results["internal_ledger"]["df"], results["internal_ledger"]["count"], results["internal_ledger"]["desc"] = \
        detect_internal_ledger_candidates(df_copy)

    results["statement_freeze"]["df"], results["statement_freeze"]["count"], results["statement_freeze"]["desc"] = \
        detect_statement_freeze_candidates(df_copy, freeze_threshold_date)

    results["cbuae_transfer"]["df"], results["cbuae_transfer"]["count"], results["cbuae_transfer"]["desc"] = \
        detect_cbuae_transfer_candidates(df_copy)  # Relies on 'Expected_Transfer_to_CB_Due'

    return results


# The `log_flag_instructions` function from your provided `compliance.py` seems fine.
# I'll include it here for completeness, assuming it's part of this file.
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
        with conn.cursor() as cursor:  # Use 'with' for cursor
            # For Azure SQL, INFORMATION_SCHEMA.TABLES is standard
            cursor.execute("SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'dormant_flags'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                # Consider creating it if init_db somehow missed it or if running standalone
                # For now, error out as per original logic
                return False, "Error: 'dormant_flags' table not found in the database. Run schema initialization."

            # For Azure SQL, to avoid inserting duplicates for a PRIMARY KEY (account_id),
            # it's better to check existence first or use MERGE if supported and preferred.
            # A simpler way for batch is to try insert and catch primary key violation, but that's per row.
            # The SELECT ... WHERE NOT EXISTS is more common in SQLite.
            # For Azure SQL, a common pattern is:
            # IF NOT EXISTS (SELECT 1 FROM dormant_flags WHERE account_id = ?)
            # BEGIN
            #    INSERT INTO dormant_flags (account_id, flag_instruction, timestamp) VALUES (?, ?, ?)
            # END
            # This needs to be executed per row or adapted for batch.

            # Simpler approach for this example (might log duplicates if account_id is not PK or Unique constraint is violated and not handled)
            # Assuming 'account_id' is PRIMARY KEY in 'dormant_flags' as per schema.py

            insert_sql = """
                         INSERT INTO dormant_flags (account_id, flag_instruction, timestamp)
                         VALUES (?, ?, ?) \
                         """
            # To handle existing:
            # MERGE dormant_flags AS target
            # USING (VALUES (?, ?, ?)) AS source (account_id, flag_instruction, timestamp)
            # ON target.account_id = source.account_id
            # WHEN NOT MATCHED THEN
            #     INSERT (account_id, flag_instruction, timestamp)
            #     VALUES (source.account_id, source.flag_instruction, source.timestamp);
            # This MERGE syntax is specific to SQL Server.

            timestamp_now = datetime.now()
            rows_inserted = 0
            skipped_due_to_existence = 0

            for acc_id in account_ids:
                if pd.notna(acc_id) and str(acc_id).strip() != '':
                    try:
                        # Check if exists first (safer for cross-db if MERGE isn't used)
                        cursor.execute("SELECT 1 FROM dormant_flags WHERE account_id = ?", (str(acc_id),))
                        if cursor.fetchone():
                            skipped_due_to_existence += 1
                            continue  # Skip if already exists

                        cursor.execute(
                            insert_sql,
                            (
                                str(acc_id),
                                f"Identified by {agent_name} for review (Threshold: {threshold_days} days) - CBUAE",
                                timestamp_now
                            )
                        )
                        rows_inserted += cursor.rowcount
                    except pyodbc.IntegrityError:  # Catches PK violation if check above somehow misses (race condition)
                        skipped_due_to_existence += 1
                    except Exception as e_row:
                        st.sidebar.warning(f"Skipping log for {acc_id} due to DB error: {e_row}")

            conn.commit()

            if skipped_due_to_existence > 0:
                return True, f"Logged {rows_inserted} new unique accounts. {skipped_due_to_existence} were already in the flagging log or caused an error."
            else:
                return True, f"Logged {rows_inserted} unique accounts for flagging review!"

    except Exception as e:
        return False, f"DB logging failed: {e}"