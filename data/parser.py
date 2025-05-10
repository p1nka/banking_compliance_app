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

        # Store the mapping between standardized and original column names
        if SESSION_COLUMN_MAPPING not in st.session_state:
            st.session_state[SESSION_COLUMN_MAPPING] = {}

        # Update the mapping with new columns
        for std, orig in zip(standardized_columns, original_columns):
            st.session_state[SESSION_COLUMN_MAPPING][std] = orig

        # Define expected columns and their types/handling
        date_cols = ['Last_Transaction_Date']
        string_cols_require_str = ["Account_ID", "Account_Type", "Account_Status", "Email_Contact_Attempt",
                                   "SMS_Contact_Attempt", "Phone_Call_Attempt", "KYC_Status", "Branch"]

        # Ensure expected columns exist, add if missing with default value 'Unknown' or NaT for date
        for col in date_cols:
            if col not in df.columns:
                df[col] = pd.NaT
                st.sidebar.warning(f"Missing expected column '{col}'. Added with missing values.")
        for col in string_cols_require_str:
            if col not in df.columns:
                df[col] = 'Unknown'
                st.sidebar.warning(f"Missing expected column '{col}'. Added with 'Unknown' values.")

        # Type conversion and cleaning for expected columns
        for col in date_cols:
            if col in df.columns:  # Check again after potentially adding
                # Show the unique values before conversion for debugging
                if not df[col].empty:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) > 0:
                        st.sidebar.info(f"Sample dates before conversion ({col}): {unique_vals[:3]}")

                # Attempt robust date conversion with multiple formats
                try:
                    # First try standard conversion with error coercing
                    df[col] = pd.to_datetime(df[col], errors='coerce')

                    # Check if we got too many NaT values (>50%)
                    if df[col].isna().mean() > 0.5:
                        st.sidebar.warning(
                            f"Over 50% of dates in '{col}' couldn't be parsed. Trying alternative formats...")

                        # Save a copy of the original column
                        orig_dates = df[col].copy()

                        # Try common date formats explicitly
                        formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                                   '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d', '%d.%m.%Y']

                        for fmt in formats:
                            try:
                                df[col] = pd.to_datetime(orig_dates, format=fmt, errors='coerce')
                                # If this format worked well (less than 25% NaT), use it
                                if df[col].isna().mean() < 0.25:
                                    st.sidebar.info(f"Successfully parsed dates using format: {fmt}")
                                    break
                            except:
                                continue
                except Exception as e:
                    st.sidebar.error(f"Error converting dates in column '{col}': {e}")
                    # Ensure column exists even if conversion failed
                    df[col] = pd.NaT

        for col in string_cols_require_str:
            if col in df.columns:  # Check again after potentially adding
                # Ensure string type and fill NaNs, strip whitespace
                try:
                    df[col] = df[col].astype(str).fillna('Unknown').str.strip()
                    # Replace common 'no data' indicators with 'Unknown'
                    df[col] = df[col].replace(['nan', 'None', '', 'Null', 'NULL', 'null'], 'Unknown', regex=True)
                except Exception as e:
                    st.sidebar.error(f"Error standardizing column '{col}': {e}")
                    # Ensure column exists with default value
                    df[col] = 'Unknown'

        # Final validation check
        if df is None or df.empty:
            st.sidebar.error("Data processing resulted in empty DataFrame. Check input data.")
            return None

        st.sidebar.success(f"✅ Data parsed and standardized successfully! Shape: {df.shape}")
        return df
    except Exception as e:
        st.sidebar.error(f"Error during data parsing/standardization: {e}")
        st.sidebar.error(f"Original columns detected: {original_columns if original_columns else 'N/A'}")
        st.sidebar.error(f"Traceback: {traceback.format_exc()}")
        return None