import pandas as pd
from datetime import datetime, timedelta
import numpy as np  # For potential numeric operations if needed

# CBUAE Dormancy Periods (examples, can be centralized in config.py)
STANDARD_INACTIVITY_YEARS = 3
PAYMENT_INSTRUMENT_UNCLAIMED_YEARS = 1
SDB_UNPAID_FEES_YEARS = 3
ELIGIBILITY_FOR_CB_TRANSFER_YEARS = 5


def check_safe_deposit_dormancy(df, report_date):
    """
    Identifies Safe Deposit Boxes meeting dormancy criteria (Art. 2.6 CBUAE).
    Criteria:
        - Account_Type == 'safe_deposit_box'
        - Charges outstanding ('yes' or numeric > 0) for > 3 years
        - No reply from tenant (values like 'no', '', 'nan', or NaN)
    Columns used:
        - Account_Type
        - SDB_Charges_Outstanding
        - Date_SDB_Charges_Became_Outstanding
        - SDB_Tenant_Communication_Received
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'SDB_Charges_Outstanding',
            'Date_SDB_Charges_Became_Outstanding', 'SDB_Tenant_Communication_Received'
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped SDB Dormancy: Missing {', '.join(missing_cols)})", {}

        # Ensure date column is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['Date_SDB_Charges_Became_Outstanding']):
            df['Date_SDB_Charges_Became_Outstanding'] = pd.to_datetime(
                df['Date_SDB_Charges_Became_Outstanding'], errors='coerce'
            )

        # Threshold is 3 years (1095 days) before report date
        threshold_date_sdb = report_date - timedelta(days=3 * 365)

        # Charges outstanding: allow "yes" or any numeric > 0
        mask_charges = (
            (df['SDB_Charges_Outstanding'].astype(str).str.lower() == "yes") |
            (pd.to_numeric(df['SDB_Charges_Outstanding'], errors='coerce').fillna(0) > 0)
        )

        # Tenant communication: allow 'no', '', 'nan', or NaN
        mask_no_reply = (
            (df['SDB_Tenant_Communication_Received'].astype(str).str.lower().isin(['no', '', 'nan'])) |
            (df['SDB_Tenant_Communication_Received'].isna())
        )

        mask = (
            (df['Account_Type'].astype(str).str.lower() == "safe_deposit_box") &
            mask_charges &
            (df['Date_SDB_Charges_Became_Outstanding'].notna()) &
            (df['Date_SDB_Charges_Became_Outstanding'] < threshold_date_sdb) &
            mask_no_reply
        )

        data = df[mask].copy()
        count = len(data)
        desc = (
            f"Safe Deposit Boxes meeting dormancy criteria (Art 2.6: >3yr unpaid, no tenant reply): {count} boxes"
        )
        details = {
            "average_outstanding_charges": pd.to_numeric(data['SDB_Charges_Outstanding'], errors='coerce').mean() if count else 0,
            "total_outstanding_charges": pd.to_numeric(data['SDB_Charges_Outstanding'], errors='coerce').sum() if count else 0,
            "earliest_charge_outstanding_date": str(data['Date_SDB_Charges_Became_Outstanding'].min().date()) if count and pd.notna(data['Date_SDB_Charges_Became_Outstanding'].min()) else "N/A",
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Safe Deposit Dormancy check: {e})", {}


def check_investment_inactivity(df, report_date):
    """
    Identifies Investment Accounts meeting dormancy criteria (Art. 2.3 CBUAE).
    Focuses on closed-ended/redeemable: No customer communication for 3 years from final maturity/redemption.
    Uses: Account_Type, Inv_Maturity_Redemption_Date, Date_Last_Customer_Communication_Any_Type
          (Implicitly assumes 'Investment_Type' might distinguish open/closed if available, otherwise broader check)
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'Inv_Maturity_Redemption_Date',
            'Date_Last_Customer_Communication_Any_Type'
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Investment Inactivity: Missing {', '.join(missing_cols)})", {}

        date_cols = ['Inv_Maturity_Redemption_Date', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        threshold_date_inv_comm = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        # Art 2.3: "no communication from the customer for a period of 3 years from final maturity or redemption date"
        data = df[
            (df['Account_Type'].astype(str).str.contains("Investment", case=False, na=False)) &
            (df['Inv_Maturity_Redemption_Date'].notna()) &
            # Condition: Maturity date is past, AND EITHER no communication ever, OR last communication was before (MaturityDate + 3 years ago from report_date effectively)
            # More accurately: check if (report_date - last_communication_date) > 3 years AND (report_date - maturity_date) > 3 years
            # AND maturity_date itself is past
            (df['Inv_Maturity_Redemption_Date'] < report_date) &  # Maturity has passed
            # Last communication is more than 3 years ago from report_date
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) | \
             (df['Date_Last_Customer_Communication_Any_Type'] < threshold_date_inv_comm)) &
            # And this period of no communication extends for at least 3 years *after* maturity
            # This means the Inv_Maturity_Redemption_Date itself must also be older than 3 years from report_date
            # if we are checking against report_date for communication.
            (df['Inv_Maturity_Redemption_Date'] < threshold_date_inv_comm)
            ].copy()

        count = len(data)
        desc = f"Investment accounts (maturing type) dormant (Art 2.3: >{STANDARD_INACTIVITY_YEARS}yr post-maturity & no recent comms): {count} accounts"
        details = {
            "earliest_maturity_date": str(data['Inv_Maturity_Redemption_Date'].min().date()) if count and pd.notna(
                data['Inv_Maturity_Redemption_Date'].min()) else "N/A",
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Investment Inactivity check: {e})", {}


def check_fixed_deposit_inactivity(df, report_date):
    """
    Identifies Fixed/Term Deposit accounts meeting dormancy criteria (Art. 2.2 CBUAE).
    Criteria: Matured, no renewal/claim request OR no customer comms if auto-renewal, for 3 years post-maturity.
    Uses: Account_Type, FTD_Maturity_Date, FTD_Auto_Renewal,
          Date_Last_FTD_Renewal_Claim_Request, Date_Last_Customer_Communication_Any_Type
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'FTD_Maturity_Date', 'FTD_Auto_Renewal',
            'Date_Last_FTD_Renewal_Claim_Request', 'Date_Last_Customer_Communication_Any_Type'
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Fixed Deposit Inactivity: Missing {', '.join(missing_cols)})", {}

        date_cols = ['FTD_Maturity_Date', 'Date_Last_FTD_Renewal_Claim_Request',
                     'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        threshold_3_years_ago = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        # Case 1: No auto-renewal
        # Art 2.2: "neither renewal nor claim request has been made in the past 3 years since the deposit matured"
        df_no_auto_renewal = df[
            (~df['FTD_Auto_Renewal'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['FTD_Maturity_Date'].notna()) &
            (df['FTD_Maturity_Date'] < report_date) &  # Matured
            # No renewal/claim in 3 years *since maturity date*.
            # So, maturity date must be at least 3 years old, and no claim since then up to maturity + 3 years.
            # More simply: (maturity date < 3_years_ago) AND (last_claim_request isNa OR last_claim_request < maturity_date + 3years)
            # Or even simpler: (maturity date < 3_years_ago) AND (last_claim_request isNA OR last_claim_request < 3_years_ago (if claim must be recent))
            # Let's stick to "no claim request in the past 3 years since deposit matured"
            (df['FTD_Maturity_Date'] < threshold_3_years_ago) &  # Mature for more than 3 years
            ((df['Date_Last_FTD_Renewal_Claim_Request'].isna()) | \
             (df['Date_Last_FTD_Renewal_Claim_Request'] < (
                         df['FTD_Maturity_Date'] + pd.DateOffset(years=STANDARD_INACTIVITY_YEARS))))

            ]

        # Case 2: With auto-renewal
        # Art 2.2: "where there is an automatic renewable clause, but there is no communication from the customer within a period of 3 years from the date of first maturity"
        # Assuming FTD_Maturity_Date is the "first maturity" or "last effective maturity"
        df_auto_renewal = df[
            (df['FTD_Auto_Renewal'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (df['FTD_Maturity_Date'].notna()) &
            (df['FTD_Maturity_Date'] < threshold_3_years_ago) &  # First/Effective maturity was >3 years ago
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) | \
             (df['Date_Last_Customer_Communication_Any_Type'] < threshold_3_years_ago))  # No comms in last 3 years
            ]

        data = pd.concat([
            df_no_auto_renewal[
                df_no_auto_renewal['Account_Type'].astype(str).str.contains("Fixed|Term", case=False, na=False)],
            df_auto_renewal[
                df_auto_renewal['Account_Type'].astype(str).str.contains("Fixed|Term", case=False, na=False)]
        ]).drop_duplicates(subset=['Account_ID']).copy()

        count = len(data)
        desc = f"Fixed Deposit accounts dormant (Art 2.2: >{STANDARD_INACTIVITY_YEARS}yr post-maturity issues): {count} accounts"
        details = {
            "earliest_maturity_date": str(data['FTD_Maturity_Date'].min().date()) if count and pd.notna(
                data['FTD_Maturity_Date'].min()) else "N/A",
            "count_auto_renewal_cases": len(
                data[data['FTD_Auto_Renewal'].astype(str).str.lower().isin(['yes', 'true', '1'])]) if count else 0,
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Fixed Deposit Inactivity check: {e})", {}


def check_demand_deposit_inactivity(df, report_date):
    """
    Identifies Demand Deposit accounts (Current, Saving, Call) meeting dormancy criteria (Art. 2.1.1 CBUAE).
    Criteria: No customer-initiated financial/non-financial transactions AND no customer communication for 3 years.
              AND customer should not have an active liability account with the same bank.
              AND (implicit from 'dormant customer' definition) address not known, no litigations.
    Uses: Account_Type, Date_Last_Cust_Initiated_Activity, Date_Last_Customer_Communication_Any_Type,
          Customer_Has_Active_Liability_Account
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'Date_Last_Cust_Initiated_Activity',
            'Date_Last_Customer_Communication_Any_Type', 'Customer_Has_Active_Liability_Account',
            # Additional from Dormant Customer Def in Art 2 (implicitly applied)
            # 'Customer_Address_Known', 'Customer_Has_Litigation_Regulatory_Reqs'
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Demand Deposit Inactivity: Missing {', '.join(missing_cols)})", {}

        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        threshold_3_years_ago = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)

        # Art 2.1.1:
        # "account where there has been no transactions (withdrawals or deposits) or non-financial actions ... for a period of 3 years"
        # AND "there has been no communication from the customer"
        # AND other dormant customer conditions from Art 2 (para after First: Dormant Accounts)
        data = df[
            (df['Account_Type'].astype(str).str.contains("Current|Saving|Call", case=False, na=False)) &
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] < threshold_3_years_ago) &  # No activity for 3 years
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) | \
             (df[
                  'Date_Last_Customer_Communication_Any_Type'] < threshold_3_years_ago)) &  # No communication for 3 years
            (~df['Customer_Has_Active_Liability_Account'].astype(str).str.lower().isin(['yes', 'true', '1']))
            # Implicitly, also other conditions for "dormant customer" (address unknown, no litigation) should be met.
            # These might be pre-filtered at a customer level or need additional flags.
            # For this function, we assume these flags ('Customer_Address_Known', 'Customer_Has_Litigation_Regulatory_Reqs') are present if strictly applying.
            # Adding them for completeness if available:
            # & (df.get('Customer_Address_Known', pd.Series(dtype=str)).astype(str).str.lower().isin(['no','false','0','nan',''])) \
            # & (df.get('Customer_Has_Litigation_Regulatory_Reqs', pd.Series(dtype=str)).astype(str).str.lower().isin(['no','false','0','nan','']))
            ].copy()

        count = len(data)
        desc = (f"Demand Deposit accounts dormant (Art 2.1.1: >{STANDARD_INACTIVITY_YEARS}yr no activity/comms, "
                f"no active liability): {count} accounts")
        details = {
            "earliest_last_activity_date": str(
                data['Date_Last_Cust_Initiated_Activity'].min().date()) if count and pd.notna(
                data['Date_Last_Cust_Initiated_Activity'].min()) else "N/A",
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Demand Deposit Inactivity check: {e})", {}


def check_unclaimed_payment_instruments(df, report_date):
    """
    Identifies unclaimed Bankers Cheques, Bank Drafts, Cashier Orders (Art. 2.4 CBUAE).
    Criteria: Unclaimed by beneficiary or customer for 1 year from issuance/trigger date,
              despite bank efforts to contact.
    Uses: Account_Type, Unclaimed_Item_Trigger_Date, Unclaimed_Item_Amount,
          (Implicitly: 'Bank_Contact_Attempted_Post_Dormancy_Trigger' or a similar flag showing bank effort)
    """
    try:
        required_columns = [
            'Account_ID', 'Account_Type', 'Unclaimed_Item_Trigger_Date', 'Unclaimed_Item_Amount',
            # 'Bank_Contact_Attempted_Post_Dormancy_Trigger' # Or a more specific "Bank_Contacted_Instrument_Holder"
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Unclaimed Payment Instruments: Missing {', '.join(missing_cols)})", {}

        if not pd.api.types.is_datetime64_dtype(df['Unclaimed_Item_Trigger_Date']):
            df['Unclaimed_Item_Trigger_Date'] = pd.to_datetime(df['Unclaimed_Item_Trigger_Date'], errors='coerce')

        threshold_1_year_ago = report_date - timedelta(days=PAYMENT_INSTRUMENT_UNCLAIMED_YEARS * 365)

        # Art 2.4: "despite the efforts of the Bank to contact the customer"
        # This implies a flag like 'Bank_Contact_Attempted_Post_Dormancy_Trigger' should be 'yes'
        # or a more specific flag for instrument contact. Let's assume it for now.
        contact_attempt_col = 'Bank_Contact_Attempted_Post_Dormancy_Trigger'  # Or more specific if available
        if contact_attempt_col not in df.columns:
            # If specific contact column not present, we might proceed but note it, or make it required.
            # For now, let's assume the check proceeds without it if missing, but ideally it's required.
            pass

        data = df[
            (df['Account_Type'].astype(str).str.contains("Bankers_Cheque|Bank_Draft|Cashier_Order", case=False,
                                                         na=False)) &
            (df['Unclaimed_Item_Trigger_Date'].notna()) &
            (df['Unclaimed_Item_Trigger_Date'] < threshold_1_year_ago) &
            (pd.to_numeric(df['Unclaimed_Item_Amount'], errors='coerce').fillna(0) > 0) &
            # Add condition for bank contact attempt if column exists
            (df.get(contact_attempt_col, pd.Series(dtype=str)).astype(str).str.lower().isin(['yes', 'true', '1']) \
                 if contact_attempt_col in df.columns else True)  # If col not present, this condition is true
            ].copy()

        count = len(data)
        desc = (f"Unclaimed payment instruments (Art 2.4: >{PAYMENT_INSTRUMENT_UNCLAIMED_YEARS}yr unclaimed, "
                f"bank contacted customer): {count} items")
        details = {
            "total_unclaimed_amount": pd.to_numeric(data['Unclaimed_Item_Amount'],
                                                    errors='coerce').sum() if count else 0,
            "earliest_trigger_date": str(data['Unclaimed_Item_Trigger_Date'].min().date()) if count and pd.notna(
                data['Unclaimed_Item_Trigger_Date'].min()) else "N/A",
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Unclaimed Payment Instruments check: {e})", {}


def check_eligible_for_cb_transfer(df, report_date):
    """
    Identifies accounts/balances eligible for transfer to Central Bank (Art. 8.1, 8.2 CBUAE).
    Accounts: Dormant 5 yrs, no other active accounts, address unknown.
    Instruments: Unclaimed 5 yrs.
    Uses: Account_Type, Date_Last_Cust_Initiated_Activity, Customer_Has_Active_Liability_Account (or similar),
          Customer_Address_Known, Unclaimed_Item_Trigger_Date
    """
    try:
        # Define flexible required columns based on type
        base_req = ['Account_ID', 'Account_Type']
        acc_specific_req = ['Date_Last_Cust_Initiated_Activity', 'Customer_Has_Active_Liability_Account',
                            'Customer_Address_Known']
        instr_specific_req = ['Unclaimed_Item_Trigger_Date']

        # For this function, we'll check if EITHER set of specific columns is mostly present alongside base
        minimal_acc_check = all(c in df.columns for c in base_req + ['Date_Last_Cust_Initiated_Activity'])
        minimal_instr_check = all(c in df.columns for c in base_req + ['Unclaimed_Item_Trigger_Date'])

        if not (minimal_acc_check or minimal_instr_check):
            return pd.DataFrame(), 0, "(Skipped CB Transfer Eligibility: Core date columns missing for both types)", {}

        # Convert date columns if they exist
        if 'Date_Last_Cust_Initiated_Activity' in df.columns and not pd.api.types.is_datetime64_dtype(
                df['Date_Last_Cust_Initiated_Activity']):
            df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'],
                                                                     errors='coerce')
        if 'Unclaimed_Item_Trigger_Date' in df.columns and not pd.api.types.is_datetime64_dtype(
                df['Unclaimed_Item_Trigger_Date']):
            df['Unclaimed_Item_Trigger_Date'] = pd.to_datetime(df['Unclaimed_Item_Trigger_Date'], errors='coerce')

        threshold_5_years_ago = report_date - timedelta(days=ELIGIBILITY_FOR_CB_TRANSFER_YEARS * 365)

        # Eligible Accounts (Art 8.1)
        eligible_accounts_list = []
        if all(c in df.columns for c in base_req + acc_specific_req):
            eligible_accounts_df_slice = df[
                (~df['Account_Type'].astype(str).str.contains("Bankers_Cheque|Bank_Draft|Cashier_Order|Safe Deposit",
                                                              case=False, na=False)) &  # Exclude instruments and SDBs
                (df['Date_Last_Cust_Initiated_Activity'].notna()) &
                (df['Date_Last_Cust_Initiated_Activity'] < threshold_5_years_ago) &
                (~df['Customer_Has_Active_Liability_Account'].astype(str).str.lower().isin(
                    ['yes', 'true', '1'])) &  # Assuming 'no other active accounts' maps here
                (df['Customer_Address_Known'].astype(str).str.lower().isin(['no', 'false', '0', 'nan', '']))
                ].copy()
            if not eligible_accounts_df_slice.empty:
                eligible_accounts_list.append(eligible_accounts_df_slice)

        # Eligible Instruments (Art 8.2)
        eligible_instruments_list = []
        if all(c in df.columns for c in base_req + instr_specific_req):
            eligible_instruments_df_slice = df[
                (df['Account_Type'].astype(str).str.contains("Bankers_Cheque|Bank_Draft|Cashier_Order", case=False,
                                                             na=False)) &
                (df['Unclaimed_Item_Trigger_Date'].notna()) &
                (df['Unclaimed_Item_Trigger_Date'] < threshold_5_years_ago)
                ].copy()
            if not eligible_instruments_df_slice.empty:
                eligible_instruments_list.append(eligible_instruments_df_slice)

        data = pd.DataFrame()
        if eligible_accounts_list or eligible_instruments_list:
            data = pd.concat(eligible_accounts_list + eligible_instruments_list).drop_duplicates(
                subset=['Account_ID']).copy()

        count = len(data)
        desc = (
            f"Accounts/instruments eligible for CBUAE transfer (Art 8: >{ELIGIBILITY_FOR_CB_TRANSFER_YEARS}yr dormancy/unclaimed "
            f"with conditions): {count} items")
        details = {
            "count_accounts_eligible": len(eligible_accounts_list[0]) if eligible_accounts_list else 0,
            "count_instruments_eligible": len(eligible_instruments_list[0]) if eligible_instruments_list else 0,
            "sample_items": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in CB Transfer Eligibility check: {e})", {}


def check_art3_process_needed(df, report_date):
    """
    Identifies accounts that are now considered dormant (based on 'Expected_Account_Dormant' flag)
    but for which the Art. 3 process (contact, 3-month wait before ledgering) has not yet been completed.
    Uses: Expected_Account_Dormant, Bank_Contact_Attempted_Post_Dormancy_Trigger, Date_Last_Bank_Contact_Attempt,
          (A flag indicating if it's already moved to internal 'dormant accounts ledger' would be useful here, e.g., 'Moved_To_Internal_Dormant_Ledger')
    """
    try:
        required_columns = [
            'Account_ID', 'Expected_Account_Dormant',
            'Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Date_Last_Bank_Contact_Attempt'
            # Add 'Moved_To_Internal_Dormant_Ledger' if available
        ]
        # Check for 'Expected_Requires_Article_3_Process' as per compliance.py
        if 'Expected_Requires_Article_3_Process' not in df.columns:
            # If this specific flag isn't there, we can infer based on Expected_Account_Dormant and contact attempts
            # but it's less precise for "requiring Art 3 process".
            # For now, let's focus on accounts that ARE dormant and bank HAS NOT YET contacted OR 3 months not passed.
            if not all(col in df.columns for col in required_columns):
                missing_cols = [col for col in required_columns if col not in df.columns]
                return pd.DataFrame(), 0, f"(Skipped Art3 Process Needed: Missing {', '.join(missing_cols)})", {}
        else:  # Expected_Requires_Article_3_Process is present
            required_columns.append('Expected_Requires_Article_3_Process')

        if 'Date_Last_Bank_Contact_Attempt' in df.columns and not pd.api.types.is_datetime64_dtype(
                df['Date_Last_Bank_Contact_Attempt']):
            df['Date_Last_Bank_Contact_Attempt'] = pd.to_datetime(df['Date_Last_Bank_Contact_Attempt'], errors='coerce')

        three_months_ago_from_report_date = report_date - timedelta(days=90)

        # Condition: Account is DORMANT (or requires Art 3 process if flag exists)
        # AND (Bank has NOT attempted contact OR (Bank attempted contact BUT 3-month wait not passed))
        # AND (implicitly) not yet moved to internal dormant ledger

        condition_dormant_or_requires_art3 = (
            df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1']))
        if 'Expected_Requires_Article_3_Process' in df.columns:
            condition_dormant_or_requires_art3 = condition_dormant_or_requires_art3 | \
                                                 (df['Expected_Requires_Article_3_Process'].astype(
                                                     str).str.lower().isin(['yes', 'true', '1']))

        data = df[
            condition_dormant_or_requires_art3 &
            (
                    (~df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                        ['yes', 'true', '1'])) |
                    (
                            (df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                                ['yes', 'true', '1'])) &
                            (df.get('Date_Last_Bank_Contact_Attempt', pd.Series(pd.NaT)).isna() | \
                             (df.get('Date_Last_Bank_Contact_Attempt',
                                     pd.Series(pd.NaT)) >= three_months_ago_from_report_date))
                    )
            )
            # Add if 'Moved_To_Internal_Dormant_Ledger' exists:
            # & (~df.get('Moved_To_Internal_Dormant_Ledger', pd.Series(dtype=str)).astype(str).str.lower().isin(['yes','true','1']))
            ].copy()

        count = len(data)
        desc = f"Dormant accounts needing/undergoing Art. 3 process (contact/wait period): {count} accounts"
        details = {
            "needs_initial_contact": len(data[~data['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(
                str).str.lower().isin(['yes', 'true', '1'])]) if count else 0,
            "in_3_month_wait_period": len(data[(data['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(
                str).str.lower().isin(['yes', 'true', '1'])) & \
                                               (data.get('Date_Last_Bank_Contact_Attempt',
                                                         pd.Series(pd.NaT)) >= three_months_ago_from_report_date)
                                               ]) if count else 0,
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Article 3 Process Needed check: {e})", {}


def check_contact_attempts_needed(df, report_date):
    """
    Identifies accounts nearing dormancy (e.g., inactive for 2.5 years but not yet 3)
    that proactively need contact attempts to prevent them from becoming dormant without due process.
    This is a proactive measure based on Art 5.
    Uses: Date_Last_Cust_Initiated_Activity, Date_Last_Customer_Communication_Any_Type,
          Bank_Contact_Attempted_Post_Dormancy_Trigger (to see if already contacted for other reasons)
    """
    try:
        required_columns = [
            'Account_ID', 'Date_Last_Cust_Initiated_Activity',
            'Date_Last_Customer_Communication_Any_Type',
            'Bank_Contact_Attempted_Post_Dormancy_Trigger'  # More general contact flag
        ]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Proactive Contact Needed: Missing {', '.join(missing_cols)})", {}

        date_cols = ['Date_Last_Cust_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type']
        for col in date_cols:
            if not pd.api.types.is_datetime64_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Example: "Nearing dormancy" could mean inactive for 2.5 years (STANDARD_INACTIVITY_YEARS - 0.5 years)
        # and not yet hit the full 3-year mark.
        # Warning period before full dormancy, e.g., 6 months before 3 years.
        # So, activity between (report_date - 3 years) and (report_date - 2.5 years)
        full_dormancy_threshold = report_date - timedelta(days=STANDARD_INACTIVITY_YEARS * 365)
        warning_threshold = report_date - timedelta(
            days=(STANDARD_INACTIVITY_YEARS * 365 - 180))  # 6 months prior to 3 years

        data = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] >= full_dormancy_threshold) &  # Not yet fully dormant by activity
            (df['Date_Last_Cust_Initiated_Activity'] < warning_threshold) &  # But in the warning period
            ((df['Date_Last_Customer_Communication_Any_Type'].isna()) | \
             (df['Date_Last_Customer_Communication_Any_Type'] < warning_threshold)) &  # And no recent comms either
            (~df['Bank_Contact_Attempted_Post_Dormancy_Trigger'].astype(str).str.lower().isin(
                ['yes', 'true', '1']))  # Bank hasn't already attempted contact recently
            # And not already flagged as Expected_Account_Dormant
            & (~df.get('Expected_Account_Dormant', pd.Series(dtype=str)).astype(str).str.lower().isin(
                ['yes', 'true', '1']))
            ].copy()

        count = len(data)
        desc = f"Accounts nearing dormancy needing proactive contact attempts (Art 5): {count} accounts"
        details = {
            "earliest_last_activity_in_warning": str(
                data['Date_Last_Cust_Initiated_Activity'].min().date()) if count and pd.notna(
                data['Date_Last_Cust_Initiated_Activity'].min()) else "N/A",
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Proactive Contact Needed check: {e})", {}


def check_high_value_dormant_accounts(df, threshold_balance=25000):
    """
    Identifies high-value dormant accounts.
    Uses: Expected_Account_Dormant, Current_Balance (assuming it's in AED or converted)
    """
    try:
        required_columns = ['Account_ID', 'Current_Balance',
                            'Expected_Account_Dormant']  # Assuming Current_Balance is AED
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped High-Value Dormant: Missing {', '.join(missing_cols)})", {}

        data = df[
            (df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])) &
            (pd.to_numeric(df['Current_Balance'], errors='coerce').fillna(0) >= threshold_balance)
            ].copy()

        count = len(data)
        desc = f"High-value dormant accounts (Balance >= AED {threshold_balance:,}): {count} accounts"
        details = {
            "total_high_value_balance_aed": pd.to_numeric(data['Current_Balance'],
                                                          errors='coerce').sum() if count else 0,
            "average_high_value_balance_aed": pd.to_numeric(data['Current_Balance'],
                                                            errors='coerce').mean() if count else 0,
            "sample_accounts": data['Account_ID'].head(3).tolist() if count else []
        }
        return data, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in High-Value Dormant check: {e})", {}


def check_dormant_to_active_transitions(df, report_date, dormant_flags_history_df=None, activity_lookback_days=30):
    """
    Identifies accounts previously flagged dormant that now show recent activity.
    Uses: Account_ID, Date_Last_Cust_Initiated_Activity, Expected_Account_Dormant
          And dormant_flags_history_df (with account_id, timestamp)
    """
    try:
        if dormant_flags_history_df is None or dormant_flags_history_df.empty or \
                not all(col in dormant_flags_history_df.columns for col in ['account_id', 'timestamp']):
            return pd.DataFrame(), 0, "(Skipped Dormant-to-Active: Dormant flag history unavailable or malformed)", {}

        required_main_cols = ['Account_ID', 'Date_Last_Cust_Initiated_Activity', 'Expected_Account_Dormant']
        if not all(col in df.columns for col in required_main_cols):
            missing_cols = [col for col in required_main_cols if col not in df.columns]
            return pd.DataFrame(), 0, f"(Skipped Dormant-to-Active: Main DF missing {', '.join(missing_cols)})", {}

        df['Date_Last_Cust_Initiated_Activity'] = pd.to_datetime(df['Date_Last_Cust_Initiated_Activity'],
                                                                 errors='coerce')
        dormant_flags_history_df['timestamp'] = pd.to_datetime(dormant_flags_history_df['timestamp'], errors='coerce')

        recent_activity_threshold_date = report_date - timedelta(days=activity_lookback_days)

        # Accounts with recent activity that are NOT currently considered dormant (per Expected_Account_Dormant)
        active_recently = df[
            (df['Date_Last_Cust_Initiated_Activity'].notna()) &
            (df['Date_Last_Cust_Initiated_Activity'] >= recent_activity_threshold_date) &
            (~df['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1']))
            ].copy()

        if active_recently.empty:
            return pd.DataFrame(), 0, f"No accounts showed recent activity and are currently active (last {activity_lookback_days} days).", {}

        # Merge with the latest dormant flag for each account
        latest_flags = dormant_flags_history_df.sort_values('timestamp', ascending=False).drop_duplicates(
            subset=['account_id'], keep='first')

        merged_df = pd.merge(active_recently, latest_flags, left_on='Account_ID', right_on='account_id', how='inner')

        # Confirm the flag was set *before* the reactivating transaction
        reactivated_confirmed = merged_df[
            merged_df['timestamp'] < merged_df['Date_Last_Cust_Initiated_Activity']].copy()

        count = len(reactivated_confirmed)
        desc = f"Dormant-to-Active transitions (activity in last {activity_lookback_days} days for previously flagged accs): {count} accounts"
        details = {
            "earliest_reactivation_date": str(
                reactivated_confirmed['Date_Last_Cust_Initiated_Activity'].min().date()) if count and pd.notna(
                reactivated_confirmed['Date_Last_Cust_Initiated_Activity'].min()) else "N/A",
            "sample_accounts": reactivated_confirmed['Account_ID'].head(3).tolist() if count else []
        }
        return reactivated_confirmed, count, desc, details
    except Exception as e:
        return pd.DataFrame(), 0, f"(Error in Dormant-to-Active check: {e})", {}


# This function replaces the old run_all_dormant_checks
def run_all_dormant_identification_checks(df, report_date_str=None, dormant_flags_history_df=None):
    """
    Runs all dormancy identification checks based on CBUAE rules and provided data schema.
    `report_date_str` should be YYYY-MM-DD.
    `dormant_flags_history_df` is needed for dormant_to_active check.
    """
    if report_date_str is None:
        report_date = datetime.now()
    else:
        try:
            report_date = datetime.strptime(report_date_str, "%Y-%m-%d")
        except ValueError:
            report_date = datetime.now()  # Fallback
            print(f"Warning: Invalid report_date_str format. Using current date: {report_date.strftime('%Y-%m-%d')}")

    df_copy = df.copy()  # Work on a copy to avoid modifying original df

    results = {
        "report_date_used": report_date.strftime("%Y-%m-%d"),
        "total_accounts_analyzed": len(df_copy),
        "sdb_dormant": {}, "investment_dormant": {}, "fixed_deposit_dormant": {}, "demand_deposit_dormant": {},
        "unclaimed_instruments": {}, "eligible_for_cb_transfer": {},
        "art3_process_needed": {}, "proactive_contact_needed": {},
        "high_value_dormant": {}, "dormant_to_active": {},
        "summary_kpis": {}  # For aggregated KPIs
    }

    # Run checks
    results["sdb_dormant"]["df"], results["sdb_dormant"]["count"], results["sdb_dormant"]["desc"], \
    results["sdb_dormant"]["details"] = \
        check_safe_deposit_dormancy(df_copy, report_date)
    results["investment_dormant"]["df"], results["investment_dormant"]["count"], results["investment_dormant"]["desc"], \
    results["investment_dormant"]["details"] = \
        check_investment_inactivity(df_copy, report_date)
    results["fixed_deposit_dormant"]["df"], results["fixed_deposit_dormant"]["count"], results["fixed_deposit_dormant"][
        "desc"], results["fixed_deposit_dormant"]["details"] = \
        check_fixed_deposit_inactivity(df_copy, report_date)
    results["demand_deposit_dormant"]["df"], results["demand_deposit_dormant"]["count"], \
    results["demand_deposit_dormant"]["desc"], results["demand_deposit_dormant"]["details"] = \
        check_demand_deposit_inactivity(df_copy, report_date)
    results["unclaimed_instruments"]["df"], results["unclaimed_instruments"]["count"], results["unclaimed_instruments"][
        "desc"], results["unclaimed_instruments"]["details"] = \
        check_unclaimed_payment_instruments(df_copy, report_date)
    results["eligible_for_cb_transfer"]["df"], results["eligible_for_cb_transfer"]["count"], \
    results["eligible_for_cb_transfer"]["desc"], results["eligible_for_cb_transfer"]["details"] = \
        check_eligible_for_cb_transfer(df_copy, report_date)
    results["art3_process_needed"]["df"], results["art3_process_needed"]["count"], results["art3_process_needed"][
        "desc"], results["art3_process_needed"]["details"] = \
        check_art3_process_needed(df_copy, report_date)
    results["proactive_contact_needed"]["df"], results["proactive_contact_needed"]["count"], \
    results["proactive_contact_needed"]["desc"], results["proactive_contact_needed"]["details"] = \
        check_contact_attempts_needed(df_copy, report_date)
    results["high_value_dormant"]["df"], results["high_value_dormant"]["count"], results["high_value_dormant"]["desc"], \
    results["high_value_dormant"]["details"] = \
        check_high_value_dormant_accounts(df_copy)  # threshold is default 25000
    results["dormant_to_active"]["df"], results["dormant_to_active"]["count"], results["dormant_to_active"]["desc"], \
    results["dormant_to_active"]["details"] = \
        check_dormant_to_active_transitions(df_copy, report_date, dormant_flags_history_df)

    # --- Aggregate Summary KPIs ---
    # Calculate total distinct dormant accounts identified by the primary dormancy checks
    # This requires an 'Expected_Account_Dormant' flag to be reliably set by these checks or a prior process.
    # For now, we'll sum counts which might lead to overcounting if one account fits multiple definitions.
    # A better way: Use the 'Expected_Account_Dormant' column if populated by these checks or a master process.
    total_dormant_identified_by_flag = 0
    total_dormant_balance_aed = 0
    if 'Expected_Account_Dormant' in df_copy.columns:
        dormant_subset = df_copy[df_copy['Expected_Account_Dormant'].astype(str).str.lower().isin(['yes', 'true', '1'])]
        total_dormant_identified_by_flag = len(dormant_subset)
        if 'Current_Balance' in dormant_subset.columns:  # Assuming Current_Balance is in AED
            total_dormant_balance_aed = pd.to_numeric(dormant_subset['Current_Balance'], errors='coerce').sum()

    results["summary_kpis"] = {
        "total_accounts_flagged_dormant": total_dormant_identified_by_flag,
        "percentage_dormant_of_total": round(
            (total_dormant_identified_by_flag / results["total_accounts_analyzed"]) * 100, 2) if results[
                                                                                                     "total_accounts_analyzed"] > 0 else 0,
        "total_dormant_balance_aed": total_dormant_balance_aed if 'Current_Balance' in df_copy.columns else "N/A (Current_Balance col missing)",
        "count_sdb_dormant": results["sdb_dormant"]["count"],
        "count_investment_dormant": results["investment_dormant"]["count"],
        "count_fixed_deposit_dormant": results["fixed_deposit_dormant"]["count"],
        "count_demand_deposit_dormant": results["demand_deposit_dormant"]["count"],
        "count_unclaimed_instruments": results["unclaimed_instruments"]["count"],
        "total_unclaimed_instruments_value": results["unclaimed_instruments"]["details"].get("total_unclaimed_amount",
                                                                                             0),
        "count_eligible_for_cb_transfer": results["eligible_for_cb_transfer"]["count"],
        "count_high_value_dormant": results["high_value_dormant"]["count"],
        "total_high_value_dormant_balance_aed": results["high_value_dormant"]["details"].get(
            "total_high_value_balance_aed", 0),
        "count_dormant_to_active_transitions": results["dormant_to_active"]["count"],
        "count_needing_art3_process": results["art3_process_needed"]["count"],
        "count_needing_proactive_contact": results["proactive_contact_needed"]["count"],
    }

    return results