import pandas as pd
import streamlit as st
from io import StringIO
import traceback
from config import SESSION_COLUMN_MAPPING


@st.cache_data(show_spinner="Parsing data...")
def parse_data(file_input):
    """Parses data, standardizes column names, converts types, and stores original names."""
    df = None
    original_columns = []
    try:
        if isinstance(file_input, pd.DataFrame):
            st.sidebar.info("Processing data from DataFrame object...")
            df = file_input.copy()
            original_columns = list(df.columns)
        elif hasattr(file_input, 'name'):
            name = file_input.name.lower()
            st.sidebar.info(f"Processing file: {name}")
            if name.endswith('.csv'):
                df = pd.read_csv(file_input)
            elif name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_input, engine='openpyxl')
            elif name.endswith('.json'):
                df = pd.read_json(file_input)
            else:
                st.sidebar.error("Unsupported file format. Please use CSV, XLSX, or JSON.")
                return None
            if df is not None: original_columns = list(df.columns)
        elif isinstance(file_input, str):  # Handle URL fetched string data
            st.sidebar.info("Processing data from URL or text string...")
            df = pd.read_csv(StringIO(file_input))  # Assuming URL content is CSV
            if df is not None: original_columns = list(df.columns)
        else:
            st.sidebar.error(f"Invalid input type for parsing: {type(file_input)}")
            return None

        if df is None:
            st.sidebar.error("Failed to read data.")
            return None
        if df.empty:
            st.sidebar.warning("The uploaded file is empty or could not be parsed into data.")
            return df

        # Debugging information
        st.sidebar.info(f"Original DataFrame shape: {df.shape}")
        st.sidebar.info(f"Original columns: {', '.join(original_columns)}")

        # Clean and standardize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_', regex=False).str.replace('[^A-Za-z0-9_]+', '',
                                                                                           regex=True)
        df.columns = [f"col_{i}" if c == "" else c for i, c in enumerate(df.columns)]  # Handle empty names
        standardized_columns = list(df.columns)

        # Map original column names to standardized names based on CBUAE regulation schema
        column_mapping = create_cbuae_column_mapping(standardized_columns)

        # Create a new DataFrame with properly mapped columns
        new_df = pd.DataFrame()

        # For each target CBUAE column, map from the standardized columns if available
        for cbuae_col, std_cols in column_mapping.items():
            mapped = False
            for std_col in std_cols:
                if std_col in df.columns:
                    new_df[cbuae_col] = df[std_col]
                    mapped = True
                    break

            # If no mapping found, add empty column with appropriate data type
            if not mapped:
                if cbuae_col.startswith('Date_'):
                    new_df[cbuae_col] = pd.NaT
                elif any(suffix in cbuae_col for suffix in ['_Amount', '_Balance', '_Outstanding']):
                    new_df[cbuae_col] = pd.NA
                else:
                    new_df[cbuae_col] = 'Unknown'

        # Ensure all required CBUAE columns exist
        required_cols = [
            'Account_ID', 'Customer_ID', 'Account_Type', 'Currency', 'Account_Creation_Date',
            'Current_Balance', 'Date_Last_Bank_Initiated_Activity', 'Date_Last_Customer_Communication_Any_Type',
            'FTD_Maturity_Date', 'FTD_Auto_Renewal', 'Date_Last_FTD_Renewal_Claim_Request',
            'Inv_Maturity_Redemption_Date', 'SDB_Charges_Outstanding', 'Date_SDB_Charges_Became_Outstanding',
            'SDB_Tenant_Communication_Received', 'Unclaimed_Item_Trigger_Date', 'Unclaimed_Item_Amount',
            'Date_Last_Cust_Initiated_Activity', 'Bank_Contact_Attempted_Post_Dormancy_Trigger',
            'Date_Last_Bank_Contact_Attempt', 'Customer_Responded_to_Bank_Contact',
            'Date_Claim_Received', 'Claim_Successful', 'Amount_Paid_on_Claim', 'Scenario_Notes',
            'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
            'Customer_Has_Litigation_Regulatory_Reqs', 'Holder_Has_Activity_On_Any_Other_Account',
            'Is_Asset_Only_Customer_Type', 'Expected_Account_Dormant', 'Expected_Requires_Article_3_Process',
            'Expected_Transfer_to_CB_Due'
        ]

        for col in required_cols:
            if col not in new_df.columns:
                if col.startswith('Date_'):
                    new_df[col] = pd.NaT
                elif any(suffix in col for suffix in ['_Amount', '_Balance', '_Outstanding']):
                    new_df[col] = pd.NA
                else:
                    new_df[col] = 'Unknown'
                st.sidebar.warning(f"Added missing required column '{col}' with default values.")

        # Properly convert column data types
        convert_column_types(new_df)

        # Store the mapping between standardized and original column names
        if SESSION_COLUMN_MAPPING not in st.session_state:
            st.session_state[SESSION_COLUMN_MAPPING] = {}

        # Update the mapping with new columns - keep track of both original to standardized
        # and standardized to CBUAE mappings
        for std, orig in zip(standardized_columns, original_columns):
            # Find which CBUAE column this standardized column maps to
            for cbuae_col, std_cols in column_mapping.items():
                if std in std_cols:
                    st.session_state[SESSION_COLUMN_MAPPING][cbuae_col] = orig
                    break
            # If not mapped to any CBUAE column, store original name
            if std not in [c for cols in column_mapping.values() for c in cols]:
                st.session_state[SESSION_COLUMN_MAPPING][std] = orig

        st.sidebar.success(f"✅ Data parsed and standardized successfully! Shape: {new_df.shape}")
        return new_df

    except Exception as e:
        st.sidebar.error(f"Error during data parsing/standardization: {e}")
        st.sidebar.error(f"Original columns detected: {original_columns if original_columns else 'N/A'}")
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        return None


def create_cbuae_column_mapping(standardized_columns):
    """
    Create mapping between standardized column names and CBUAE schema column names.
    This mapping helps identify which standardized columns correspond to which CBUAE columns.
    """
    # Define mapping with potential matches for each CBUAE column
    # For each CBUAE column, list possible standardized column names
    # (ordered by preference)
    return {
        'Account_ID': ['Account_ID', 'AccountID', 'Account_Number', 'AccountNumber', 'AcctID', 'ID'],
        'Customer_ID': ['Customer_ID', 'CustomerID', 'Client_ID', 'ClientID', 'CustID'],
        'Account_Type': ['Account_Type', 'AccountType', 'Type', 'Product', 'Account_Category'],
        'Currency': ['Currency', 'Ccy', 'AccountCurrency', 'CurrencyCode'],
        'Account_Creation_Date': ['Account_Creation_Date', 'CreationDate', 'OpenDate', 'Open_Date', 'Created_Date'],
        'Current_Balance': ['Current_Balance', 'Balance', 'AccountBalance', 'BalanceAmount'],
        'Date_Last_Bank_Initiated_Activity': ['Date_Last_Bank_Initiated_Activity', 'Last_Bank_Activity',
                                              'BankActivityDate'],
        'Date_Last_Customer_Communication_Any_Type': ['Date_Last_Customer_Communication', 'LastCommunication',
                                                      'Last_Contact_Date'],
        'FTD_Maturity_Date': ['FTD_Maturity_Date', 'MaturityDate', 'FixedDepositMaturity', 'Maturity'],
        'FTD_Auto_Renewal': ['FTD_Auto_Renewal', 'AutoRenewal', 'Auto_Renew', 'Is_Auto_Renewal'],
        'Date_Last_FTD_Renewal_Claim_Request': ['Date_Last_FTD_Renewal', 'RenewalRequestDate', 'Last_Renewal_Date'],
        'Inv_Maturity_Redemption_Date': ['Inv_Maturity_Date', 'InvestmentMaturity', 'RedemptionDate'],
        'SDB_Charges_Outstanding': ['SDB_Charges', 'SafeDepositCharges', 'OutstandingCharges'],
        'Date_SDB_Charges_Became_Outstanding': ['SDB_Charges_Date', 'ChargesOutstandingDate'],
        'SDB_Tenant_Communication_Received': ['SDB_Tenant_Communication', 'TenantResponse', 'Has_Tenant_Responded'],
        'Unclaimed_Item_Trigger_Date': ['Unclaimed_Trigger_Date', 'UnclaimedDate', 'ItemTriggerDate'],
        'Unclaimed_Item_Amount': ['Unclaimed_Amount', 'UnclaimedValue', 'ItemAmount'],
        'Date_Last_Cust_Initiated_Activity': ['Last_Transaction_Date', 'Date_Last_Transaction', 'LastActivityDate',
                                              'LastTxnDate'],
        'Bank_Contact_Attempted_Post_Dormancy_Trigger': ['Contact_Attempted', 'BankContactAttempt', 'ContactTried'],
        'Date_Last_Bank_Contact_Attempt': ['Last_Contact_Attempt', 'ContactAttemptDate', 'AttemptDate'],
        'Customer_Responded_to_Bank_Contact': ['Customer_Responded', 'ResponseReceived', 'Has_Responded'],
        'Date_Claim_Received': ['Claim_Date', 'ClaimReceived', 'DateClaimed'],
        'Claim_Successful': ['Claim_Success', 'Is_Claim_Successful', 'ClaimResult'],
        'Amount_Paid_on_Claim': ['Claim_Amount', 'AmountPaid', 'PaymentAmount'],
        'Scenario_Notes': ['Notes', 'Comments', 'Scenario', 'Description'],
        'Customer_Address_Known': ['Address_Known', 'Has_Address', 'KnownAddress'],
        'Customer_Has_Active_Liability_Account': ['Has_Active_Account', 'Active_Liability', 'HasLiabilityAccount'],
        'Customer_Has_Litigation_Regulatory_Reqs': ['Has_Litigation', 'RegulatoryRequirements', 'HasLegalRestriction'],
        'Holder_Has_Activity_On_Any_Other_Account': ['Has_Other_Activity', 'Activity_Other_Account',
                                                     'OtherAccountActivity'],
        'Is_Asset_Only_Customer_Type': ['Is_Asset_Only', 'Asset_Only_Customer', 'AssetOnlyType'],
        'Expected_Account_Dormant': ['Account_Status', 'Is_Dormant', 'Dormant', 'StatusDormant'],
        'Expected_Requires_Article_3_Process': ['Requires_Art3', 'Article_3_Required', 'Art3Process'],
        'Expected_Transfer_to_CB_Due': ['Transfer_To_CB', 'CBTransfer', 'CentralBankTransfer']
    }


def convert_column_types(df):
    """Convert DataFrame columns to appropriate types based on CBUAE schema."""
    # Convert date columns
    date_columns = [col for col in df.columns if col.startswith('Date_')]
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.sidebar.warning(f"Could not convert {col} to datetime: {e}")

    # Convert numeric columns (amounts and balances)
    numeric_columns = [col for col in df.columns if
                       any(suffix in col for suffix in ['_Amount', '_Balance', '_Outstanding'])]
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                st.sidebar.warning(f"Could not convert {col} to numeric: {e}")

    # Convert Yes/No columns to standardized format
    boolean_columns = [
        'FTD_Auto_Renewal', 'SDB_Tenant_Communication_Received',
        'Bank_Contact_Attempted_Post_Dormancy_Trigger', 'Customer_Responded_to_Bank_Contact',
        'Claim_Successful', 'Customer_Address_Known', 'Customer_Has_Active_Liability_Account',
        'Customer_Has_Litigation_Regulatory_Reqs', 'Holder_Has_Activity_On_Any_Other_Account',
        'Is_Asset_Only_Customer_Type', 'Expected_Account_Dormant',
        'Expected_Requires_Article_3_Process', 'Expected_Transfer_to_CB_Due'
    ]

    for col in boolean_columns:
        if col in df.columns:
            try:
                # Convert various boolean representations to 'Yes'/'No'
                df[col] = df[col].astype(str).str.lower()
                df[col] = df[col].replace({
                    'true': 'Yes', 'yes': 'Yes', 'y': 'Yes', '1': 'Yes', 't': 'Yes',
                    'false': 'No', 'no': 'No', 'n': 'No', '0': 'No', 'f': 'No',
                    'nan': 'Unknown', 'none': 'Unknown', '': 'Unknown',
                    'na': 'Unknown', 'null': 'Unknown', 'unknown': 'Unknown'
                })
            except Exception as e:
                st.sidebar.warning(f"Could not standardize {col} values: {e}")

    # Ensure Account_Type values are standardized
    if 'Account_Type' in df.columns:
        try:
            df['Account_Type'] = df['Account_Type'].astype(str).str.lower()
            # Map various account type terminology to standard values
            df['Account_Type'] = df['Account_Type'].replace({
                'current': 'Current', 'currentaccount': 'Current', 'current_account': 'Current',
                'saving': 'Savings', 'savings': 'Savings', 'savingsaccount': 'Savings', 'savings_account': 'Savings',
                'fd': 'Fixed Deposit', 'fixed': 'Fixed Deposit', 'fixeddeposit': 'Fixed Deposit',
                'term': 'Fixed Deposit',
                'call': 'Call', 'callaccount': 'Call', 'call_account': 'Call',
                'inv': 'Investment', 'invest': 'Investment', 'investaccount': 'Investment',
                'sdb': 'Safe Deposit', 'safe': 'Safe Deposit', 'safebox': 'Safe Deposit'
            })
        except Exception as e:
            st.sidebar.warning(f"Could not standardize Account_Type values: {e}")