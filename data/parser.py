"""
Data parsing utilities for the banking compliance application.
"""
import pandas as pd
import streamlit as st
from datetime import datetime
import json
from io import StringIO


@st.cache_data
def parse_data(data_source):
    """
    Parse data from various sources (file, URL response, DataFrame).
    
    Args:
        data_source: Can be uploaded file, URL response text, or DataFrame
        
    Returns:
        pd.DataFrame: Parsed and standardized DataFrame
    """
    try:
        df = None
        
        # Handle different data source types
        if isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
        elif hasattr(data_source, 'name'):  # Uploaded file
            if data_source.name.endswith('.csv'):
                df = pd.read_csv(data_source)
            elif data_source.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_source)
            elif data_source.name.endswith('.json'):
                df = pd.read_json(data_source)
        elif isinstance(data_source, str):  # URL response or JSON string
            if data_source.strip().startswith('{') or data_source.strip().startswith('['):
                # JSON string
                df = pd.read_json(StringIO(data_source))
            else:
                # CSV string
                df = pd.read_csv(StringIO(data_source))
        
        if df is None or df.empty:
            st.error("Could not parse data or data is empty")
            return pd.DataFrame()
        
        # Standardize column names and data
        df_standardized = standardize_dataframe(df)
        
        st.success(f"✅ Successfully parsed {len(df_standardized)} rows with {len(df_standardized.columns)} columns")
        return df_standardized
        
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return pd.DataFrame()


def standardize_dataframe(df):
    """
    Standardize DataFrame column names and data types for banking compliance.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: Standardized DataFrame
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    standardized_df = df.copy()
    
    # Column mapping from various possible names to standard names
    column_mapping = {
        # Account Information
        'account_id': 'Account_ID',
        'accountid': 'Account_ID',
        'account_number': 'Account_ID',
        'acc_id': 'Account_ID',
        
        'customer_id': 'Customer_ID',
        'customerid': 'Customer_ID',
        'cust_id': 'Customer_ID',
        'client_id': 'Customer_ID',
        
        'account_type': 'Account_Type',
        'accounttype': 'Account_Type',
        'acc_type': 'Account_Type',
        'type': 'Account_Type',
        
        'currency': 'Currency',
        'curr': 'Currency',
        'ccy': 'Currency',
        
        # Balance Information
        'balance': 'Current_Balance',
        'current_balance': 'Current_Balance',
        'currentbalance': 'Current_Balance',
        'amount': 'Current_Balance',
        'balance_amount': 'Current_Balance',
        
        # Date Fields
        'last_activity': 'Date_Last_Cust_Initiated_Activity',
        'last_activity_date': 'Date_Last_Cust_Initiated_Activity',
        'date_last_activity': 'Date_Last_Cust_Initiated_Activity',
        'lastactivity': 'Date_Last_Cust_Initiated_Activity',
        
        'creation_date': 'Account_Creation_Date',
        'account_creation_date': 'Account_Creation_Date',
        'created_date': 'Account_Creation_Date',
        'date_created': 'Account_Creation_Date',
        
        # Communication Dates
        'last_communication': 'Date_Last_Customer_Communication_Any_Type',
        'last_comm_date': 'Date_Last_Customer_Communication_Any_Type',
        'communication_date': 'Date_Last_Customer_Communication_Any_Type',
        
        # Status Fields
        'dormant': 'Expected_Account_Dormant',
        'is_dormant': 'Expected_Account_Dormant',
        'dormant_flag': 'Expected_Account_Dormant',
        'expected_dormant': 'Expected_Account_Dormant',
        
        'address_known': 'Customer_Address_Known',
        'address_available': 'Customer_Address_Known',
        'has_address': 'Customer_Address_Known',
        
        'has_liability': 'Customer_Has_Active_Liability_Account',
        'liability_account': 'Customer_Has_Active_Liability_Account',
        'active_liability': 'Customer_Has_Active_Liability_Account',
    }
    
    # Apply column mapping
    current_columns = [col.lower().replace(' ', '_').replace('-', '_') for col in standardized_df.columns]
    new_column_names = {}
    
    for i, col in enumerate(standardized_df.columns):
        normalized_col = current_columns[i]
        if normalized_col in column_mapping:
            new_column_names[col] = column_mapping[normalized_col]
    
    # Rename columns
    standardized_df = standardized_df.rename(columns=new_column_names)
    
    # Store mapping in session state for reference
    if 'SESSION_COLUMN_MAPPING' not in st.session_state:
        st.session_state['SESSION_COLUMN_MAPPING'] = {}
    
    # Reverse mapping for display purposes
    reverse_mapping = {v: k for k, v in new_column_names.items()}
    st.session_state['SESSION_COLUMN_MAPPING'].update(reverse_mapping)
    
    # Standardize data types
    standardized_df = standardize_data_types(standardized_df)
    
    # Add missing required columns with default values
    required_columns = {
        'Expected_Account_Dormant': 'No',
        'Expected_Requires_Article_3_Process': 'No',
        'Expected_Transfer_to_CB_Due': 'No',
        'Customer_Address_Known': 'Unknown',
        'Customer_Has_Active_Liability_Account': 'Unknown',
    }
    
    for col, default_value in required_columns.items():
        if col not in standardized_df.columns:
            standardized_df[col] = default_value
    
    return standardized_df


def standardize_data_types(df):
    """
    Standardize data types for banking compliance fields.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with standardized data types
    """
    if df.empty:
        return df
    
    # Date columns to convert
    date_columns = [
        'Account_Creation_Date',
        'Date_Last_Cust_Initiated_Activity',
        'Date_Last_Customer_Communication_Any_Type',
        'Date_Last_Bank_Contact_Attempt',
        'FTD_Maturity_Date',
        'Inv_Maturity_Redemption_Date',
        'Date_SDB_Charges_Became_Outstanding',
        'Unclaimed_Item_Trigger_Date',
        'Date_Claim_Received',
    ]
    
    # Convert date columns
    for col in date_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert {col} to datetime: {e}")
    
    # Numeric columns to convert
    numeric_columns = [
        'Current_Balance',
        'SDB_Charges_Outstanding',
        'Unclaimed_Item_Amount',
        'Amount_Paid_on_Claim',
    ]
    
    # Convert numeric columns
    for col in numeric_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert {col} to numeric: {e}")
    
    # Boolean-like columns to standardize
    boolean_columns = [
        'Expected_Account_Dormant',
        'Expected_Requires_Article_3_Process', 
        'Expected_Transfer_to_CB_Due',
        'FTD_Auto_Renewal',
        'SDB_Tenant_Communication_Received',
        'Bank_Contact_Attempted_Post_Dormancy_Trigger',
        'Customer_Responded_to_Bank_Contact',
        'Claim_Successful',
        'Customer_Address_Known',
        'Customer_Has_Active_Liability_Account',
        'Customer_Has_Litigation_Regulatory_Reqs',
        'Holder_Has_Activity_On_Any_Other_Account',
        'Is_Asset_Only_Customer_Type',
    ]
    
    # Standardize boolean-like columns
    for col in boolean_columns:
        if col in df.columns:
            try:
                df[col] = df[col].astype(str).str.lower()
                df[col] = df[col].replace({
                    'true': 'Yes', '1': 'Yes', 'yes': 'Yes', 'y': 'Yes',
                    'false': 'No', '0': 'No', 'no': 'No', 'n': 'No',
                    'nan': 'Unknown', 'none': 'Unknown', '': 'Unknown'
                })
                # Set default for any remaining NaN values
                df[col] = df[col].fillna('Unknown')
            except Exception as e:
                st.warning(f"Could not standardize {col}: {e}")
    
    return df


def validate_required_fields(df):
    """
    Validate that required fields are present for compliance analysis.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        tuple: (is_valid, missing_fields)
    """
    required_fields = [
        'Account_ID',
        'Customer_ID', 
        'Account_Type',
    ]
    
    missing_fields = [field for field in required_fields if field not in df.columns]
    
    return len(missing_fields) == 0, missing_fields


def create_sample_data():
    """
    Create sample banking data for testing.
    
    Returns:
        pd.DataFrame: Sample DataFrame
    """
    import random
    from datetime import timedelta
    
    sample_data = []
    account_types = ['Savings', 'Current', 'Fixed Deposit', 'Investment']
    currencies = ['AED', 'USD', 'EUR']
    
    for i in range(50):
        last_activity = datetime.now() - timedelta(days=random.randint(30, 2000))
        
        record = {
            'Account_ID': f'ACC{i+1:06d}',
            'Customer_ID': f'CUST{(i//3)+1:05d}',
            'Account_Type': random.choice(account_types),
            'Currency': random.choice(currencies),
            'Current_Balance': round(random.uniform(100, 100000), 2),
            'Date_Last_Cust_Initiated_Activity': last_activity,
            'Expected_Account_Dormant': 'Yes' if (datetime.now() - last_activity).days > 1095 else 'No',
            'Customer_Address_Known': random.choice(['Yes', 'No', 'Unknown']),
            'Customer_Has_Active_Liability_Account': random.choice(['Yes', 'No']),
        }
        sample_data.append(record)
    
    return pd.DataFrame(sample_data)
