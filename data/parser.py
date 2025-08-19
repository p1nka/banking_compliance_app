# --- UPDATED parser.py for your exact CSV structure ---

"""
Data parsing utilities for the banking compliance application.
Updated to handle your exact CSV column structure.
"""
import pandas as pd
import streamlit as st
from datetime import datetime
import json
from io import StringIO

# UPDATED COLUMN MAPPING for your exact CSV structure
# Since your CSV already has the correct column names, we'll use a direct mapping
COLUMN_MAPPING = {
    # Customer Information - exact matches from your CSV
    'customer_id': 'customer_id',
    'customer_type': 'customer_type',
    'full_name_en': 'full_name_en',
    'full_name_ar': 'full_name_ar',
    'id_number': 'id_number',
    'id_type': 'id_type',
    'date_of_birth': 'date_of_birth',
    'nationality': 'nationality',

    # Address and Contact - exact matches
    'address_line1': 'address_line1',
    'address_line2': 'address_line2',
    'city': 'city',
    'emirate': 'emirate',
    'country': 'country',
    'postal_code': 'postal_code',
    'phone_primary': 'phone_primary',
    'phone_secondary': 'phone_secondary',
    'email_primary': 'email_primary',
    'email_secondary': 'email_secondary',
    'address_known': 'address_known',
    'last_contact_date': 'last_contact_date',
    'last_contact_method': 'last_contact_method',

    # KYC and Risk - exact matches
    'kyc_status': 'kyc_status',
    'kyc_expiry_date': 'kyc_expiry_date',
    'risk_rating': 'risk_rating',

    # Core Account Information - exact matches
    'account_id': 'account_id',
    'account_type': 'account_type',
    'account_subtype': 'account_subtype',
    'account_name': 'account_name',
    'currency': 'currency',
    'account_status': 'account_status',
    'dormancy_status': 'dormancy_status',

    # Important Date Fields - exact matches
    'opening_date': 'opening_date',
    'closing_date': 'closing_date',
    'last_transaction_date': 'last_transaction_date',
    'last_system_transaction_date': 'last_system_transaction_date',
    'maturity_date': 'maturity_date',
    'last_statement_date': 'last_statement_date',

    # Balance Information - exact matches
    'balance_current': 'balance_current',
    'balance_available': 'balance_available',
    'balance_minimum': 'balance_minimum',
    'interest_rate': 'interest_rate',
    'interest_accrued': 'interest_accrued',

    # Account Details - exact matches
    'is_joint_account': 'is_joint_account',
    'joint_account_holders': 'joint_account_holders',
    'has_outstanding_facilities': 'has_outstanding_facilities',
    'auto_renewal': 'auto_renewal',
    'statement_frequency': 'statement_frequency',

    # Process Tracking Fields - exact matches (CRITICAL for dormancy analysis)
    'tracking_id': 'tracking_id',
    'dormancy_trigger_date': 'dormancy_trigger_date',
    'dormancy_period_start': 'dormancy_period_start',
    'dormancy_period_months': 'dormancy_period_months',
    'dormancy_classification_date': 'dormancy_classification_date',
    'transfer_eligibility_date': 'transfer_eligibility_date',
    'current_stage': 'current_stage',
    'contact_attempts_made': 'contact_attempts_made',
    'last_contact_attempt_date': 'last_contact_attempt_date',
    'waiting_period_start': 'waiting_period_start',
    'waiting_period_end': 'waiting_period_end',
    'transferred_to_ledger_date': 'transferred_to_ledger_date',
    'transferred_to_cb_date': 'transferred_to_cb_date',
    'cb_transfer_amount': 'cb_transfer_amount',
    'cb_transfer_reference': 'cb_transfer_reference',
    'exclusion_reason': 'exclusion_reason',

    # System Fields - exact matches
    'created_date': 'created_date',
    'updated_date': 'updated_date',
    'updated_by': 'updated_by'
}

# Standardized mapping for compatibility with existing code
# This creates aliases for commonly expected field names
STANDARDIZED_ALIASES = {
    # Common aliases for dormancy agents
    'Account_ID': 'account_id',
    'Customer_ID': 'customer_id',
    'Account_Type': 'account_type',
    'Current_Balance': 'balance_current',
    'Date_Last_Cust_Initiated_Activity': 'last_transaction_date',
    'Expected_Account_Dormant': 'dormancy_status',
    'Contact_Attempts_Made': 'contact_attempts_made',
    'Customer_Type': 'customer_type',
    'Maturity_Date': 'maturity_date',
    'Account_Status': 'account_status',
    'Customer_Address_Known': 'address_known',
    'Customer_Has_Active_Liability_Account': 'has_outstanding_facilities'
}


@st.cache_data
def parse_data(data_source):
    """
    Parse data from various sources (file, URL response, DataFrame).
    Updated for your exact CSV structure.
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

        # Log original structure
        st.sidebar.info(f"Original data: {len(df)} rows, {len(df.columns)} columns")

        # Process the DataFrame
        df_processed = standardize_dataframe(df)

        st.success(f"✅ Successfully parsed {len(df_processed)} rows with {len(df_processed.columns)} columns")
        return df_processed

    except Exception as e:
        st.error(f"Error parsing data: {e}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return pd.DataFrame()


def standardize_dataframe(df):
    """
    Standardize DataFrame - UPDATED for your CSV structure.
    Since your CSV already has correct column names, we mostly preserve them.
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying original
    standardized_df = df.copy()

    # Clean column names (remove extra spaces, etc.)
    standardized_df.columns = [col.strip() for col in standardized_df.columns]

    # Log what we found
    st.sidebar.write("**Original Columns Found:**")
    st.sidebar.write(f"Total: {len(standardized_df.columns)}")

    # Show key columns for dormancy analysis
    key_dormancy_columns = [
        'account_id', 'account_type', 'account_subtype', 'dormancy_status',
        'last_transaction_date', 'balance_current', 'contact_attempts_made',
        'maturity_date', 'current_stage', 'transfer_eligibility_date'
    ]

    found_key_columns = []
    missing_key_columns = []

    for col in key_dormancy_columns:
        if col in standardized_df.columns:
            found_key_columns.append(col)
        else:
            missing_key_columns.append(col)

    if found_key_columns:
        st.sidebar.success(f"✅ Found {len(found_key_columns)} key dormancy columns")

    if missing_key_columns:
        st.sidebar.warning(f"⚠️ Missing {len(missing_key_columns)} key columns: {missing_key_columns}")

    # Add standardized aliases for backward compatibility
    for alias, original in STANDARDIZED_ALIASES.items():
        if original in standardized_df.columns and alias not in standardized_df.columns:
            standardized_df[alias] = standardized_df[original]

    # Standardize data types
    standardized_df = standardize_data_types(standardized_df)

    # Add any missing required columns with defaults
    required_columns = {
        'Expected_Requires_Article_3_Process': 'No',
        'Expected_Transfer_to_CB_Due': 'No',
        'Customer_Address_Known': standardized_df.get('address_known', 'Unknown'),
        'Customer_Has_Active_Liability_Account': standardized_df.get('has_outstanding_facilities', 'Unknown'),
    }

    # Only add if they don't exist
    for col, default_value in required_columns.items():
        if col not in standardized_df.columns:
            if isinstance(default_value, str) and default_value in standardized_df.columns:
                # Use existing column if available
                standardized_df[col] = standardized_df[default_value]
            else:
                # Use default value
                standardized_df[col] = default_value

    return standardized_df


def standardize_data_types(df):
    """
    Standardize data types for banking compliance fields - UPDATED for your CSV format.
    """
    if df.empty:
        return df

    # Date columns from your CSV (in YYYY-MM-DD format)
    date_cols = [
        'date_of_birth',
        'last_contact_date',
        'kyc_expiry_date',
        'opening_date',
        'closing_date',
        'last_transaction_date',
        'last_system_transaction_date',
        'maturity_date',
        'last_statement_date',
        'dormancy_trigger_date',
        'dormancy_period_start',
        'dormancy_classification_date',
        'transfer_eligibility_date',
        'last_contact_attempt_date',
        'waiting_period_start',
        'waiting_period_end',
        'transferred_to_ledger_date',
        'transferred_to_cb_date',
        'created_date',
        'updated_date'
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
        'id_number',
        'postal_code',
        'phone_primary',
        'phone_secondary',
        'balance_current',
        'balance_available',
        'balance_minimum',
        'interest_rate',
        'interest_accrued',
        'joint_account_holders',
        'dormancy_period_months',
        'contact_attempts_made',
        'cb_transfer_amount',
        'cb_transfer_reference'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # String/Categorical columns - ensure consistent formatting
    string_cols = [
        'customer_type', 'account_type', 'account_subtype', 'currency',
        'account_status', 'dormancy_status', 'current_stage', 'kyc_status',
        'risk_rating', 'is_joint_account', 'has_outstanding_facilities',
        'auto_renewal', 'address_known'
    ]

    for col in string_cols:
        if col in df.columns:
            # Convert to string and handle NaN values
            df[col] = df[col].astype(str).replace('nan', '').replace('NaN', '')

            # Standardize common values
            if col in ['dormancy_status']:
                df[col] = df[col].str.upper()
            elif col in ['account_type', 'account_subtype', 'current_stage']:
                df[col] = df[col].str.upper()
            elif col in ['is_joint_account', 'has_outstanding_facilities', 'auto_renewal']:
                # Standardize Yes/No values
                df[col] = df[col].str.upper().replace({
                    'YES': 'Yes', 'NO': 'No', 'TRUE': 'Yes', 'FALSE': 'No',
                    '1': 'Yes', '0': 'No', 'Y': 'Yes', 'N': 'No'
                }).fillna('Unknown')

    return df


def validate_required_fields(df):
    """Validate that required fields are present for compliance analysis."""
    required_fields = ['account_id', 'account_type', 'balance_current']
    missing_fields = [field for field in required_fields if field not in df.columns]
    return not missing_fields, missing_fields


def create_sample_data():
    """Create sample banking data for testing - matching your CSV structure."""
    import random
    from datetime import timedelta

    sample_data = []
    account_types = ['CURRENT', 'SAVINGS', 'FIXED_DEPOSIT', 'INVESTMENT']
    account_subtypes = ['PERSONAL', 'PREMIUM', 'REGULAR', 'BUSINESS', 'ISLAMIC', 'VIP']
    currencies = ['AED', 'USD', 'EUR']

    for i in range(50):
        last_activity = datetime.now() - timedelta(days=random.randint(30, 2000))
        sample_data.append({
            'account_id': f'ACC{i + 1:06d}',
            'customer_id': f'CUS{(i // 3) + 1:05d}',
            'customer_type': random.choice(['INDIVIDUAL', 'CORPORATE']),
            'account_type': random.choice(account_types),
            'account_subtype': random.choice(account_subtypes),
            'currency': random.choice(currencies),
            'balance_current': round(random.uniform(1000, 500000), 2),
            'last_transaction_date': last_activity.strftime('%Y-%m-%d'),
            'dormancy_status': 'DORMANT' if (datetime.now() - last_activity).days > 1095 else 'ACTIVE',
            'contact_attempts_made': random.randint(0, 5),
            'current_stage': random.choice(['FLAGGED', 'WAITING', 'TRANSFER_READY']),
            'account_status': 'ACTIVE',
            'maturity_date': (datetime.now() + timedelta(days=random.randint(30, 365))).strftime(
                '%Y-%m-%d') if random.choice([True, False]) else '',
            'opening_date': (datetime.now() - timedelta(days=random.randint(365, 3000))).strftime('%Y-%m-%d'),
            'created_date': datetime.now().strftime('%Y-%m-%d'),
            'updated_date': datetime.now().strftime('%Y-%m-%d'),
            'updated_by': 'SYSTEM'
        })
    return pd.DataFrame(sample_data)


def debug_column_mapping(df_original):
    """
    Debug function to show how columns are being mapped.
    Updated for your CSV structure.
    """
    print("COLUMN MAPPING DEBUG FOR YOUR CSV")
    print("=" * 40)
    print(f"Original CSV has {len(df_original.columns)} columns")
    print(f"Original columns: {list(df_original.columns)}")

    # Since your CSV already has correct names, show direct mapping
    print("\nDirect Column Usage (no mapping needed):")
    print("-" * 30)

    key_columns = [
        'account_id', 'account_type', 'account_subtype', 'dormancy_status',
        'last_transaction_date', 'balance_current', 'contact_attempts_made',
        'maturity_date', 'current_stage', 'transfer_eligibility_date'
    ]

    found_count = 0
    for col in key_columns:
        if col in df_original.columns:
            print(f"✅ {col} → FOUND")
            found_count += 1
        else:
            print(f"❌ {col} → NOT FOUND")

    print(f"\nMapping Summary:")
    print(f"- Total key columns: {len(key_columns)}")
    print(f"- Found columns: {found_count}")
    print(f"- Missing columns: {len(key_columns) - found_count}")

    # Show data types
    print(f"\nData Types Preview:")
    for col in key_columns[:5]:  # Show first 5
        if col in df_original.columns:
            dtype = df_original[col].dtype
            sample = df_original[col].iloc[0] if not df_original[col].empty else 'Empty'
            print(f"- {col}: {dtype} (sample: {sample})")

    return {}  # No mapping needed since columns are already correct


def test_parser_with_your_csv():
    """
    Test function specifically for your CSV structure.
    """
    print("PARSER TEST FOR YOUR SPECIFIC CSV")
    print("=" * 40)

    expected_columns = [
        'account_id', 'customer_id', 'account_type', 'account_subtype',
        'balance_current', 'last_transaction_date', 'dormancy_status',
        'contact_attempts_made', 'current_stage', 'maturity_date'
    ]

    print("Expected columns in your CSV:")
    for col in expected_columns:
        print(f"  ✅ {col}")

    print("\nData type expectations:")
    print("  account_id: string")
    print("  account_type: string (CURRENT, SAVINGS, FIXED_DEPOSIT, INVESTMENT)")
    print("  account_subtype: string (PERSONAL, PREMIUM, etc.)")
    print("  dormancy_status: string (DORMANT)")
    print("  last_transaction_date: date (YYYY-MM-DD format)")
    print("  balance_current: float")
    print("  contact_attempts_made: integer")
    print("  current_stage: string (FLAGGED, WAITING, TRANSFER_READY)")

    return expected_columns