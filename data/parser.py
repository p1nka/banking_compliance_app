import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO
import streamlit as st
from config import SESSION_COLUMN_MAPPING


# Caching the parse function to improve performance on repeated calls
@st.cache_data
def parse_data(data_source):
    """
    Parse uploaded data into a standardized format for the application.

    Args:
        data_source: File upload, DataFrame, or string data

    Returns:
        DataFrame with standardized columns
    """
    # Convert the data source to a DataFrame
    df = None
    try:
        if isinstance(data_source, pd.DataFrame):
            # Already a DataFrame, make a copy
            df = data_source.copy()
        elif hasattr(data_source, 'read'):
            # File upload
            file_extension = data_source.name.split('.')[-1].lower()

            if file_extension == 'csv':
                df = pd.read_csv(data_source)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(data_source)
            elif file_extension == 'json':
                df = pd.read_json(data_source)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        elif isinstance(data_source, str):
            # String data (probably from URL)
            df = pd.read_csv(StringIO(data_source))
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")
        raise e

    # Ensure essential columns exist
    required_columns = [
        "Account_ID", "Customer_ID", "Account_Type",
        "Date_Last_Cust_Initiated_Activity", "Expected_Account_Dormant"
    ]

    # Check if required columns exist (case-insensitive)
    df_columns_lower = [col.lower() for col in df.columns]
    column_mapping = {}

    # Map columns by case-insensitive matching
    for req_col in required_columns:
        req_col_lower = req_col.lower()
        found = False

        for i, col in enumerate(df_columns_lower):
            if col == req_col_lower:
                # Found a match, use the original case from the uploaded data
                orig_col = df.columns[i]
                if orig_col != req_col:
                    # Rename only if different
                    column_mapping[orig_col] = req_col
                found = True
                break

        if not found:
            # Add missing required column with default values
            st.warning(f"Required column '{req_col}' not found. Adding with default values.")
            if req_col == "Account_ID":
                df[req_col] = [f"ACC{i + 1000}" for i in range(len(df))]
            elif req_col == "Customer_ID":
                df[req_col] = [f"CUST{i + 1000}" for i in range(len(df))]
            elif req_col == "Account_Type":
                df[req_col] = "Unknown"
            elif req_col == "Date_Last_Cust_Initiated_Activity":
                df[req_col] = datetime.now().strftime("%Y-%m-%d")
            elif req_col == "Expected_Account_Dormant":
                df[req_col] = "No"

    # Rename columns based on mapping
    if column_mapping:
        df = df.rename(columns=column_mapping)

    # Ensure date columns are properly formatted
    date_columns = [
        "Date_Last_Cust_Initiated_Activity",
        "Account_Open_Date",
        "Last_Communication_Date"
    ]

    for date_col in date_columns:
        if date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df[date_col] = df[date_col].dt.strftime('%Y-%m-%d')
            except Exception as e:
                st.warning(f"Could not convert {date_col} to date format. Error: {e}")

    # Standardize yes/no fields
    boolean_columns = [
        "Expected_Account_Dormant",
        "Has_Address",
        "Has_Active_Accounts"
    ]

    for bool_col in boolean_columns:
        if bool_col in df.columns:
            # Convert to string first
            df[bool_col] = df[bool_col].astype(str).str.lower()
            # Standardize to Yes/No
            df[bool_col] = df[bool_col].apply(
                lambda x: "Yes" if x.lower() in ["yes", "y", "true", "1", "t"] else
                ("No" if x.lower() in ["no", "n", "false", "0", "f"] else x)
            )

    # Store original column names for display purposes
    st.session_state[SESSION_COLUMN_MAPPING] = {col: col for col in df.columns}

    # Log the parsing results
    st.sidebar.info(f"Processed {len(df)} rows with {len(df.columns)} columns")

    return df