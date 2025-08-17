# --- START OF FILE parser.py ---

"""
Data parsing utilities for the banking compliance application.
"""
import pandas as pd
import streamlit as st
from datetime import datetime
import json
from io import StringIO

# CUSTOM MAPPING FOR YOUR SPECIFIC CSV STRUCTURE
# Based on analysis of your exact CSV columns and data format
COLUMN_MAPPING = {
    # Core Account Information
    'account_id': 'Account_ID',
    'customer_id': 'Customer_ID',
    'account_type': 'Account_Type',
    'account_subtype': 'Account_Subtype',
    'account_name': 'Account_Name',
    'currency': 'Currency',
    'account_status': 'Account_Status',

    # Customer Information
    'customer_type': 'Customer_Type',
    'full_name_en': 'Customer_Name_English',
    'full_name_ar': 'Customer_Name_Arabic',
    'id_number': 'Customer_ID_Number',
    'id_type': 'Customer_ID_Type',
    'date_of_birth': 'Customer_Date_of_Birth',
    'nationality': 'Customer_Nationality',

    # Address and Contact
    'address_line1': 'Customer_Address_Line1',
    'address_line2': 'Customer_Address_Line2',
    'city': 'Customer_City',
    'emirate': 'Customer_Emirate',
    'country': 'Customer_Country',
    'postal_code': 'Customer_Postal_Code',
    'phone_primary': 'Customer_Phone_Primary',
    'phone_secondary': 'Customer_Phone_Secondary',
    'email_primary': 'Customer_Email_Primary',
    'email_secondary': 'Customer_Email_Secondary',
    'address_known': 'Customer_Address_Known',

    # Balance Information
    'balance_current': 'Current_Balance',
    'balance_available': 'Available_Balance',
    'balance_minimum': 'Minimum_Balance',
    'interest_rate': 'Interest_Rate',
    'interest_accrued': 'Interest_Accrued',

    # Account Details
    'is_joint_account': 'Is_Joint_Account',
    'joint_account_holders': 'Joint_Account_Holders_Count',
    'has_outstanding_facilities': 'Has_Outstanding_Facilities',
    'auto_renewal': 'Auto_Renewal',
    'statement_frequency': 'Statement_Frequency',

    # Critical Date Fields for Dormancy Analysis
    'last_transaction_date': 'Date_Last_Cust_Initiated_Activity',
    'last_system_transaction_date': 'Date_Last_System_Activity',
    'opening_date': 'Account_Creation_Date',
    'closing_date': 'Account_Closing_Date',
    'maturity_date': 'Maturity_Date',
    'last_statement_date': 'Date_Last_Statement',

    # Communication and Contact
    'last_contact_date': 'Date_Last_Customer_Communication_Any_Type',
    'last_contact_method': 'Last_Contact_Method',
    'last_contact_attempt_date': 'Date_Last_Contact_Attempt',
    'contact_attempts_made': 'Contact_Attempts_Made',

    # KYC and Risk
    'kyc_status': 'KYC_Status',
    'kyc_expiry_date': 'KYC_Expiry_Date',
    'risk_rating': 'Risk_Rating',

    # Dormancy and Process Status (CRITICAL MAPPINGS)
    'dormancy_status': 'Expected_Account_Dormant',
    'dormancy_trigger_date': 'Dormancy_Trigger_Date',
    'dormancy_period_start': 'Dormancy_Period_Start',
    'dormancy_period_months': 'Dormancy_Period_Months',
    'dormancy_classification_date': 'Dormancy_Classification_Date',
    'transfer_eligibility_date': 'Transfer_Eligibility_Date',
    'current_stage': 'Current_Stage',

    # Transfer and Process Tracking
    'waiting_period_start': 'Waiting_Period_Start',
    'waiting_period_end': 'Waiting_Period_End',
    'transferred_to_ledger_date': 'Transferred_to_Ledger_Date',
    'transferred_to_cb_date': 'Transferred_to_CB_Date',
    'cb_transfer_amount': 'CB_Transfer_Amount',
    'cb_transfer_reference': 'CB_Transfer_Reference',
    'exclusion_reason': 'Exclusion_Reason',

    # System Fields
    'tracking_id': 'Tracking_ID',
    'created_date': 'Created_Date',
    'updated_date': 'Updated_Date',
    'updated_by': 'Updated_By',
}


@st.cache_data
def parse_data(data_source):
    """
    Parse data from various sources (file, URL response, DataFrame).
    """
    try:
        df = None
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        elif hasattr(data_source, 'name'):  # Uploaded file
            if data_source.name.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_source)
            elif data_source.name.endswith('.json'):
                df = pd.read_json(data_source)
        elif isinstance(data_source, str):  # String data
            try:
                # Attempt to parse as JSON first
                df = pd.read_json(StringIO(data_source))
            except ValueError:
                # Fallback to CSV if JSON parsing fails
                df = pd.read_csv(StringIO(data_source))

        if df is None or df.empty:
            st.error("Could not parse data or data is empty")
            return pd.DataFrame()

        df_standardized = standardize_dataframe(df)
        st.success(f"✅ Successfully parsed {len(df_standardized)} rows with {len(df_standardized.columns)} columns")
        return df_standardized

    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return pd.DataFrame()


def standardize_dataframe(df):
    """Standardize DataFrame column names and data types."""
    if df.empty:
        return df

    standardized_df = df.copy()
    current_columns = [col.lower().replace(' ', '_').replace('-', '_') for col in standardized_df.columns]
    new_column_names = {}

    for i, col in enumerate(standardized_df.columns):
        normalized_col = current_columns[i]
        if normalized_col in COLUMN_MAPPING:
            new_column_names[col] = COLUMN_MAPPING[normalized_col]

    standardized_df = standardized_df.rename(columns=new_column_names)

    standardized_df = standardize_data_types(standardized_df)

    # FIXED: Only add required columns if they don't exist, and don't override existing data
    required_columns = {
        'Expected_Requires_Article_3_Process': 'No',
        'Expected_Transfer_to_CB_Due': 'No',
        'Customer_Address_Known': 'Unknown',
        'Customer_Has_Active_Liability_Account': 'Unknown',
    }

    # CRITICAL FIX: Don't override Expected_Account_Dormant if it already exists
    if 'Expected_Account_Dormant' not in standardized_df.columns:
        required_columns['Expected_Account_Dormant'] = 'No'

    for col, default_value in required_columns.items():
        if col not in standardized_df.columns:
            standardized_df[col] = default_value

    return standardized_df


def standardize_data_types(df):
    """Standardize data types for banking compliance fields - CUSTOMIZED FOR YOUR CSV."""
    if df.empty: return df

    # Date columns from your CSV (exact format: YYYY-MM-DD)
    date_cols = [
        'Date_Last_Cust_Initiated_Activity',  # last_transaction_date
        'Date_Last_System_Activity',  # last_system_transaction_date
        'Account_Creation_Date',  # opening_date
        'Account_Closing_Date',  # closing_date
        'Maturity_Date',  # maturity_date
        'Date_Last_Statement',  # last_statement_date
        'Date_Last_Customer_Communication_Any_Type',  # last_contact_date
        'Date_Last_Contact_Attempt',  # last_contact_attempt_date
        'KYC_Expiry_Date',  # kyc_expiry_date
        'Customer_Date_of_Birth',  # date_of_birth
        'Dormancy_Trigger_Date',  # dormancy_trigger_date
        'Dormancy_Period_Start',  # dormancy_period_start
        'Dormancy_Classification_Date',  # dormancy_classification_date
        'Transfer_Eligibility_Date',  # transfer_eligibility_date
        'Waiting_Period_Start',  # waiting_period_start
        'Waiting_Period_End',  # waiting_period_end
        'Transferred_to_Ledger_Date',  # transferred_to_ledger_date
        'Transferred_to_CB_Date',  # transferred_to_cb_date
        'Created_Date',  # created_date
        'Updated_Date'  # updated_date
    ]

    for col in date_cols:
        if col in df.columns:
            try:
                # Your CSV uses YYYY-MM-DD format consistently
                df[col] = pd.to_datetime(df[col], format='%Y-%m-%d', errors='coerce')
            except Exception:
                try:
                    # Fallback for any other date formats
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass  # Keep as string if conversion fails

    # Numeric columns from your CSV
    numeric_cols = [
        'Current_Balance',  # balance_current
        'Available_Balance',  # balance_available
        'Minimum_Balance',  # balance_minimum
        'Interest_Rate',  # interest_rate
        'Interest_Accrued',  # interest_accrued
        'Contact_Attempts_Made',  # contact_attempts_made
        'Joint_Account_Holders_Count',  # joint_account_holders
        'Dormancy_Period_Months',  # dormancy_period_months
        'CB_Transfer_Amount',  # cb_transfer_amount
        'Customer_ID_Number',  # id_number
        'Customer_Postal_Code',  # postal_code
        'Customer_Phone_Primary',  # phone_primary
        'Customer_Phone_Secondary'  # phone_secondary
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Boolean/Yes-No columns from your CSV
    yes_no_cols = [
        'Is_Joint_Account',  # is_joint_account (NO/YES)
        'Has_Outstanding_Facilities',  # has_outstanding_facilities (NO/YES)
        'Auto_Renewal'  # auto_renewal (if applicable)
    ]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper().replace({
                'YES': 'Yes', 'NO': 'No', 'TRUE': 'Yes', 'FALSE': 'No',
                '1': 'Yes', '0': 'No', 'Y': 'Yes', 'N': 'No'
            }).fillna('Unknown')

    # Address Known column (specific format: NO/YES)
    if 'Customer_Address_Known' in df.columns:
        df['Customer_Address_Known'] = df['Customer_Address_Known'].astype(str).str.upper().replace({
            'NO': 'No', 'YES': 'Yes', 'UNKNOWN': 'Unknown'
        }).fillna('Unknown')

    # CRITICAL: Dormancy Status - Your CSV has 'DORMANT' values
    if 'Expected_Account_Dormant' in df.columns:
        df['Expected_Account_Dormant'] = df['Expected_Account_Dormant'].astype(str).str.upper().replace({
            'DORMANT': 'Yes',  # Your CSV value
            'ACTIVE': 'No',  # Possible value
            'INACTIVE': 'Yes',  # Possible value
            'YES': 'Yes',
            'NO': 'No',
            'TRUE': 'Yes',
            'FALSE': 'No'
        }).fillna('Unknown')

    # Account Status standardization
    if 'Account_Status' in df.columns:
        # Keep original values but ensure consistency
        df['Account_Status'] = df['Account_Status'].astype(str).str.upper()

    # Current Stage standardization
    if 'Current_Stage' in df.columns:
        # Keep original values (FLAGGED, etc.)
        df['Current_Stage'] = df['Current_Stage'].astype(str).str.upper()

    # Risk Rating standardization
    if 'Risk_Rating' in df.columns:
        # Keep original values (LOW, MEDIUM, HIGH)
        df['Risk_Rating'] = df['Risk_Rating'].astype(str).str.upper()

    # KYC Status standardization
    if 'KYC_Status' in df.columns:
        # Keep original values (VALID, etc.)
        df['KYC_Status'] = df['KYC_Status'].astype(str).str.upper()

    return df


def validate_required_fields(df):
    """Validate that required fields are present for compliance analysis."""
    required_fields = ['Account_ID', 'Customer_ID', 'Account_Type']
    missing_fields = [field for field in required_fields if field not in df.columns]
    return not missing_fields, missing_fields


def create_sample_data():
    """Create sample banking data for testing."""
    import random
    from datetime import timedelta
    sample_data = []
    for i in range(50):
        last_activity = datetime.now() - timedelta(days=random.randint(30, 2000))
        sample_data.append({
            'Account_ID': f'ACC{i + 1:06d}',
            'Customer_ID': f'CUST{(i // 3) + 1:05d}',
            'Account_Type': random.choice(['Savings', 'Current', 'Fixed Deposit']),
            'Currency': random.choice(['AED', 'USD', 'EUR']),
            'Current_Balance': round(random.uniform(100, 100000), 2),
            'Date_Last_Cust_Initiated_Activity': last_activity,
            'Expected_Account_Dormant': 'Yes' if (datetime.now() - last_activity).days > 1095 else 'No'
        })
    return pd.DataFrame(sample_data)


def debug_column_mapping(df_original):
    """
    Debug function to show how columns are being mapped.
    Call this function to troubleshoot parsing issues.
    """
    print("COLUMN MAPPING DEBUG FOR YOUR CSV")
    print("=" * 40)
    print(f"Original CSV has {len(df_original.columns)} columns")
    print(f"Original columns: {list(df_original.columns)}")

    current_columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_original.columns]
    new_column_names = {}
    mapped_count = 0

    print("\nColumn Mapping Results:")
    print("-" * 30)

    for i, col in enumerate(df_original.columns):
        normalized_col = current_columns[i]
        if normalized_col in COLUMN_MAPPING:
            new_column_names[col] = COLUMN_MAPPING[normalized_col]
            print(f"✅ {col} → {COLUMN_MAPPING[normalized_col]}")
            mapped_count += 1
        else:
            print(f"❌ {col} → (no mapping - will keep original)")

    print(f"\nMapping Summary:")
    print(f"- Total columns: {len(df_original.columns)}")
    print(f"- Mapped columns: {mapped_count}")
    print(f"- Unmapped columns: {len(df_original.columns) - mapped_count}")

    print(f"\nCritical mappings for dormancy analysis:")
    critical_mappings = [
        ('account_id', 'Account_ID'),
        ('balance_current', 'Current_Balance'),
        ('last_transaction_date', 'Date_Last_Cust_Initiated_Activity'),
        ('dormancy_status', 'Expected_Account_Dormant'),
        ('maturity_date', 'Maturity_Date'),
        ('contact_attempts_made', 'Contact_Attempts_Made')
    ]

    for original, expected in critical_mappings:
        if original in [col.lower() for col in df_original.columns]:
            print(f"✅ {original} found and will be mapped to {expected}")
        else:
            print(f"❌ {original} NOT FOUND in CSV")

    return new_column_names


def test_parser_with_your_csv():
    """
    Test function specifically for your CSV structure.
    This will show exactly how your data gets transformed.
    """
    # This would be called with your actual CSV data
    print("PARSER TEST FOR YOUR SPECIFIC CSV")
    print("=" * 40)

    # Expected transformations for your data:
    expected_transformations = {
        'account_id': 'Account_ID',
        'customer_id': 'Customer_ID',
        'account_type': 'Account_Type',
        'balance_current': 'Current_Balance',
        'last_transaction_date': 'Date_Last_Cust_Initiated_Activity',
        'dormancy_status': 'Expected_Account_Dormant',
        'maturity_date': 'Maturity_Date',
        'contact_attempts_made': 'Contact_Attempts_Made',
        'account_status': 'Account_Status',
        'current_stage': 'Current_Stage'
    }

    print("Key transformations that will happen:")
    for csv_col, standard_col in expected_transformations.items():
        print(f"  {csv_col} → {standard_col}")

    print("\nData type conversions:")
    print("  dormancy_status: 'DORMANT' → Expected_Account_Dormant: 'Yes'")
    print("  last_transaction_date: '2022-03-28' → Date_Last_Cust_Initiated_Activity: datetime")
    print("  balance_current: 118555.13 → Current_Balance: float")
    print("  contact_attempts_made: 0 → Contact_Attempts_Made: int")

    return expected_transformations