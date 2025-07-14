import streamlit as st
import requests
import pandas as pd
from io import StringIO
import time
from datetime import datetime, timedelta

from config import (
    DB_SERVER, DB_NAME, DB_PORT,
    SESSION_APP_DF, SESSION_DATA_PROCESSED, SESSION_COLUMN_MAPPING,
    DEFAULT_DORMANT_DAYS, DEFAULT_FREEZE_DAYS, DEFAULT_CBUAE_DATE
)

# Try to import data parser, create fallback if not available
try:
    from data.parser import parse_data
except ImportError:
    def parse_data(data_source):
        """Fallback parser function"""
        if isinstance(data_source, pd.DataFrame):
            return data_source
        elif hasattr(data_source, 'name'):  # Uploaded file
            if data_source.name.endswith('.csv'):
                return pd.read_csv(data_source)
            elif data_source.name.endswith(('.xlsx', '.xls')):
                return pd.read_excel(data_source)
        elif isinstance(data_source, str):  # CSV string
            return pd.read_csv(StringIO(data_source))
        return pd.DataFrame()

from database.connection import get_db_connection
from database.operations import save_to_db


def render_sidebar():
    """Render the main sidebar with upload options and settings."""
    st.sidebar.header("📤 Data Upload")

    # Upload method selection
    upload_method = st.sidebar.radio(
        "Select upload method:",
        [
            "**Upload File (CSV/XLSX/JSON)**",
            "**Upload via URL**",
            "**Load Data from Azure SQL Database**"
        ],
        key="upload_method_radio"
    )

    uploaded_data_source = handle_upload_method(upload_method)

    # Process button
    process_button_disabled = uploaded_data_source is None
    process_clicked = st.sidebar.button(
        "Process Uploaded/Fetched Data",
        key="process_data_button",
        disabled=process_button_disabled
    )

    if process_clicked and uploaded_data_source is not None:
        process_uploaded_data(uploaded_data_source)

    # Display app modes if data is processed
    if st.session_state.get(SESSION_DATA_PROCESSED, False):
        render_app_modes()
        render_compliance_thresholds()

    # Add report date and agent name settings
    render_analysis_settings()


def handle_upload_method(upload_method):
    """Handle the selected upload method and return the uploaded data source."""
    uploaded_data_source = None

    if upload_method == "**Upload File (CSV/XLSX/JSON)**":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Account Dataset",
            type=["csv", "xlsx", "xls", "json"],
            key="data_file_uploader"
        )
        if uploaded_file:
            uploaded_data_source = uploaded_file
            st.sidebar.caption(f"Selected: {uploaded_file.name}")

    elif upload_method == "**Upload via URL**":
        url_input = st.sidebar.text_input("Enter CSV file URL:", key="url_input")
        
        # Show URL validation info
        if url_input:
            if url_input.startswith(('http://', 'https://')):
                st.sidebar.info("✅ Valid URL format")
            else:
                st.sidebar.warning("⚠️ URL should start with http:// or https://")
        
        # Check if data was already fetched from URL
        if "url_fetched_data" in st.session_state and st.session_state.get("last_url") == url_input:
            uploaded_data_source = st.session_state["url_fetched_data"]
            st.sidebar.success("✅ URL data ready for processing")
            # Show preview of fetched data
            try:
                preview_lines = uploaded_data_source.split('\n')[:3]
                with st.sidebar.expander("📋 Data Preview"):
                    for line in preview_lines:
                        st.text(line[:80] + "..." if len(line) > 80 else line)
            except:
                pass
        else:
            # Fetch button
            if st.sidebar.button("Fetch Data from URL", key="fetch_url_button"):
                if url_input:
                    if not url_input.startswith(('http://', 'https://')):
                        st.sidebar.error("❌ Please enter a valid URL starting with http:// or https://")
                    else:
                        try:
                            with st.spinner("⏳ Fetching data from URL..."):
                                # Add headers to mimic browser request
                                headers = {
                                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                                }
                                
                                response = requests.get(url_input, timeout=30, headers=headers)
                                response.raise_for_status()
                                
                                # Check if response is actually CSV-like data
                                content = response.text
                                
                                # Basic validation for CSV content
                                if not content.strip():
                                    st.sidebar.error("❌ URL returned empty content")
                                elif len(content.split('\n')) < 2:
                                    st.sidebar.error("❌ URL content doesn't appear to be CSV (less than 2 lines)")
                                elif ',' not in content.split('\n')[0]:
                                    st.sidebar.warning("⚠️ Warning: First line doesn't contain commas. May not be CSV format.")
                                
                                # Store fetched data in session state
                                st.session_state["url_fetched_data"] = content
                                st.session_state["last_url"] = url_input
                                
                                # Show success with data info
                                lines = content.split('\n')
                                total_lines = len([line for line in lines if line.strip()])
                                st.sidebar.success(f"✅ Fetched! {total_lines} lines of data")
                                
                                # Show preview
                                with st.sidebar.expander("📋 Data Preview"):
                                    # Show first few lines
                                    preview_lines = [line for line in lines[:5] if line.strip()]
                                    for i, line in enumerate(preview_lines):
                                        display_line = line[:80] + "..." if len(line) > 80 else line
                                        if i == 0:
                                            st.text(f"Header: {display_line}")
                                        else:
                                            st.text(f"Row {i}: {display_line}")
                                    
                                    if total_lines > 5:
                                        st.text(f"... and {total_lines - 5} more lines")
                                
                                # Set uploaded_data_source for processing
                                uploaded_data_source = content
                                
                        except requests.exceptions.Timeout:
                            st.sidebar.error("❌ Request timed out. Try a different URL or check your connection.")
                        except requests.exceptions.ConnectionError:
                            st.sidebar.error("❌ Connection error. Check the URL and your internet connection.")
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code == 404:
                                st.sidebar.error("❌ URL not found (404). Please check the URL.")
                            elif e.response.status_code == 403:
                                st.sidebar.error("❌ Access forbidden (403). The URL may require authentication.")
                            else:
                                st.sidebar.error(f"❌ HTTP Error {e.response.status_code}: {e}")
                        except requests.exceptions.RequestException as e:
                            st.sidebar.error(f"❌ URL Fetch Error: {e}")
                        except Exception as e:
                            st.sidebar.error(f"❌ Error processing URL data: {e}")
                else:
                    st.sidebar.warning("⚠️ Please enter a valid URL")

        # Additional help for URL upload
        with st.sidebar.expander("💡 URL Upload Tips"):
            st.markdown("""
            **Supported URLs:**
            - Direct links to CSV files
            - GitHub raw file URLs
            - Google Sheets export URLs
            - Any publicly accessible CSV endpoint
            
            **Example URLs:**
            - `https://raw.githubusercontent.com/user/repo/main/data.csv`
            - `https://docs.google.com/spreadsheets/d/.../export?format=csv`
            - `https://example.com/data/accounts.csv`
            
            **Requirements:**
            - Must be publicly accessible (no authentication)
            - Should return CSV format data
            - File size should be reasonable (<100MB)
            """)

    elif upload_method == "**Load Data from Azure SQL Database**":
        # Check if data was loaded from SQL
        if "sql_loaded_data" in st.session_state:
            uploaded_data_source = st.session_state["sql_loaded_data"]
            st.sidebar.info(f"✅ SQL data ready: {len(uploaded_data_source)} rows")
            # Clear the temporary storage
            del st.session_state["sql_loaded_data"]
        else:
            # Show the SQL upload interface
            uploaded_data_source = handle_sql_upload()

    return uploaded_data_source


def handle_sql_upload():
    """Handle loading data from Azure SQL Database."""
    st.sidebar.subheader("Azure SQL Data Loader")
    
    # Connection status indicator
    conn = get_db_connection()
    if conn:
        st.sidebar.success("✅ Database Connected")
    else:
        st.sidebar.error("❌ Database Connection Failed")
        return None
    
    # Get available tables
    try:
        if hasattr(conn, 'execute'):
            # SQLAlchemy engine
            with conn.connect() as test_conn:
                tables_result = test_conn.execute("""
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                    ORDER BY TABLE_NAME
                """)
                available_tables = [row[0] for row in tables_result]
        else:
            # Direct connection
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """)
            available_tables = [row[0] for row in cursor.fetchall()]
            cursor.close()
            
        if not available_tables:
            st.sidebar.warning("No tables found in database")
            return None
            
    except Exception as e:
        st.sidebar.error(f"Error fetching tables: {e}")
        return None
    
    # Table selector with row counts
    st.sidebar.write("**Available Tables:**")
    
    # Show table info
    table_info = {}
    for table in available_tables:
        try:
            if hasattr(conn, 'execute'):
                with conn.connect() as test_conn:
                    count_result = test_conn.execute(f"SELECT COUNT(*) FROM [{table}]")
                    row_count = count_result.fetchone()[0]
            else:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                row_count = cursor.fetchone()[0]
                cursor.close()
            table_info[table] = row_count
        except:
            table_info[table] = "Error"
    
    # Display table options with row counts
    table_options = []
    for table in available_tables:
        count = table_info[table]
        if isinstance(count, int):
            table_options.append(f"{table} ({count:,} rows)")
        else:
            table_options.append(f"{table} (unknown rows)")
    
    selected_table_option = st.sidebar.selectbox(
        "Select table to load:",
        table_options,
        key="table_selector"
    )
    
    # Extract actual table name
    selected_table = selected_table_option.split(" (")[0] if selected_table_option else None
    
    # Record limit selector
    record_limit = st.sidebar.selectbox(
        "Records to load:",
        [100, 500, 1000, 5000, "All"],
        index=2,  # Default to 1000
        key="record_limit_selector"
    )
    
    # Query preview
    if selected_table:
        if record_limit == "All":
            query_preview = f"SELECT * FROM [{selected_table}]"
        else:
            query_preview = f"SELECT TOP {record_limit} * FROM [{selected_table}]"
        
        with st.sidebar.expander("Query Preview"):
            st.code(query_preview, language="sql")
    
    # Load button
    if st.sidebar.button("Load from Database", key="load_db_button"):
        if selected_table:
            return load_data_from_database(selected_table, record_limit)
        else:
            st.sidebar.error("Please select a table")
            return None
    
    return None


def load_data_from_database(table_name, record_limit=1000):
    """Load data from database table and return as DataFrame."""
    
    with st.spinner(f"⏳ Loading data from {table_name}..."):
        try:
            conn = get_db_connection()
            if conn is None:
                st.sidebar.error("❌ Database connection failed")
                return None

            # Build query based on record limit
            if record_limit == "All":
                query = f"SELECT * FROM [{table_name}]"
            else:
                query = f"SELECT TOP {record_limit} * FROM [{table_name}]"

            st.sidebar.info(f"Executing: {query}")
            
            # Execute query and load into DataFrame
            if hasattr(conn, 'execute'):
                # SQLAlchemy engine
                df = pd.read_sql(query, conn)
            else:
                # Direct connection (pymssql/pyodbc)
                df = pd.read_sql(query, conn)

            if df.empty:
                st.sidebar.warning(f"Table '{table_name}' has no data.")
                return None
            
            # Show success message with data info
            st.sidebar.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns from '{table_name}'")
            
            # Show preview in sidebar
            with st.sidebar.expander("📋 Data Preview"):
                st.dataframe(df.head(3), use_container_width=True)
                
                # Show column info
                st.write("**Columns:**")
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    st.write(f"- {col}: {dtype} ({non_null}/{len(df)} non-null)")
            
            # Store loaded data in session state for immediate processing
            st.session_state["sql_loaded_data"] = df
            st.session_state["sql_source_table"] = table_name
            
            return df

        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {e}")
            st.sidebar.exception(e)  # Show full traceback for debugging
            return None


def process_uploaded_data(uploaded_data_source):
    """Enhanced process_uploaded_data to handle SQL data properly."""
    
    # Initialize session state variables if they don't exist
    if SESSION_COLUMN_MAPPING not in st.session_state:
        st.session_state[SESSION_COLUMN_MAPPING] = {}
    
    with st.spinner("⏳ Processing and standardizing data..."):
        progress_text = st.sidebar.empty()
        progress_text.info("Starting data processing...")

        try:
            # Handle different data source types
            if isinstance(uploaded_data_source, pd.DataFrame):
                # Data is already a DataFrame (from SQL load)
                data_copy = uploaded_data_source.copy()
                progress_text.info("Processing SQL DataFrame...")
                source_info = f"SQL table: {st.session_state.get('sql_source_table', 'Unknown')}"
                
            else:
                # Data from file upload or URL
                data_copy = uploaded_data_source
                progress_text.info("Parsing uploaded data...")
                source_info = "Uploaded file/URL"

            # Parse/standardize the data
            progress_text.info("Standardizing data...")
            df_parsed = parse_data(data_copy)
            
        except Exception as e:
            st.sidebar.error(f"Error processing data: {e}")
            progress_text.error("Data processing failed")
            return

    if df_parsed is not None and not df_parsed.empty:
        # Data was successfully parsed
        st.session_state[SESSION_APP_DF] = df_parsed
        st.session_state[SESSION_DATA_PROCESSED] = True

        progress_text.success(f"✓ Processed {len(df_parsed)} rows, {len(df_parsed.columns)} columns")

        # Show data source info
        st.sidebar.info(f"📊 Data Source: {source_info}")

        # Update chat message if available
        try:
            from config import SESSION_CHAT_MESSAGES
            
            # Safely get column mapping
            column_mapping = st.session_state.get(SESSION_COLUMN_MAPPING, {})
            
            # Create column info safely
            cols_info = []
            for col in df_parsed.columns[:5]:  # Limit to first 5 columns
                orig_name = column_mapping.get(col, col)
                cols_info.append(f"`{orig_name}`")

            if cols_info:
                cols_example = ', '.join(cols_info)
                initial_message = (
                    f"Data ({len(df_parsed)} rows) from {source_info} processed! "
                    f"Columns include: {cols_example}...\n"
                    f"You can now use the analysis modes."
                )
            else:
                initial_message = f"Data ({len(df_parsed)} rows) from {source_info} processed successfully!"

            st.session_state[SESSION_CHAT_MESSAGES] = [
                {"role": "assistant", "content": initial_message}
            ]
        except ImportError:
            # SESSION_CHAT_MESSAGES not available, skip
            pass

        st.sidebar.success("✅ Data processed successfully!")
        
        # Show processing summary
        with st.sidebar.expander("📈 Processing Summary"):
            st.write(f"**Source:** {source_info}")
            st.write(f"**Rows:** {len(df_parsed):,}")
            st.write(f"**Columns:** {len(df_parsed.columns)}")
            
            # Show data types
            st.write("**Data Types:**")
            dtype_counts = df_parsed.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"- {dtype}: {count} columns")
        
        st.balloons()

    elif df_parsed is not None and df_parsed.empty:
        st.sidebar.error("Data parsing resulted in empty dataset.")
        st.session_state[SESSION_DATA_PROCESSED] = False
        st.session_state[SESSION_APP_DF] = None
    else:
        st.sidebar.error("❌ Data parsing failed.")
        st.session_state[SESSION_DATA_PROCESSED] = False
        st.session_state[SESSION_APP_DF] = None

    # Clear progress text
    progress_text.empty()
    
    # Rerun to update the UI
    st.rerun()


def render_app_modes():
    """Render the application mode selector in the sidebar."""
    st.sidebar.header("🚀 Analysis Modes")
    st.session_state.app_mode = st.sidebar.selectbox(
        "Select Application Mode",
        [
            "🏦 Dormant Account Analyzer",
            "🔒 Dormant Compliance Analyzer",
            "💬 IA Chat",
            "🔍 SQL Bot"
        ],
        key="app_mode_selector"
    )


def render_analysis_settings():
    """Render analysis settings in the sidebar."""
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Report date for dormancy analysis
    report_date = st.sidebar.date_input(
        "Report Date (Dormancy)", 
        st.session_state.get("report_date_for_dormancy", datetime.now().date()),
        key="report_date_input"
    )
    st.session_state.report_date_for_dormancy = report_date

    # Agent name for compliance logging
    agent_name = st.sidebar.text_input(
        "Agent Name (Compliance)", 
        st.session_state.get("agent_name_for_compliance", "SystemAuditor"),
        key="agent_name_input"
    )
    st.session_state.agent_name_for_compliance = agent_name


def render_compliance_thresholds():
    """Render the compliance thresholds in the sidebar."""
    st.sidebar.subheader("⏱️ Compliance Thresholds")

    # General inactivity threshold for flagging accounts as dormant
    general_inactivity_threshold_days = st.sidebar.number_input(
        "Flagging Inactivity Threshold (days)",
        min_value=1,
        value=st.session_state.get("flagging_inactivity_threshold_days", DEFAULT_DORMANT_DAYS),
        step=30,
        key="flag_threshold_days"
    )
    
    # Calculate and display the threshold date
    general_inactivity_threshold_date = datetime.now() - timedelta(days=general_inactivity_threshold_days)
    st.sidebar.caption(f"Flagging Threshold: {general_inactivity_threshold_date.strftime('%Y-%m-%d')}")

    # Freeze threshold for accounts that have been dormant for a long time
    freeze_inactivity_threshold_days = st.sidebar.number_input(
        "Freeze Inactivity Threshold (days)",
        min_value=1,
        value=st.session_state.get("freeze_inactivity_threshold_days", DEFAULT_FREEZE_DAYS),
        step=30,
        key="freeze_threshold_days"
    )
    
    freeze_inactivity_threshold_date = datetime.now() - timedelta(days=freeze_inactivity_threshold_days)
    st.sidebar.caption(f"Freeze Threshold: {freeze_inactivity_threshold_date.strftime('%Y-%m-%d')}")

    # CBUAE transfer cutoff date
    cbuae_cutoff_str = st.sidebar.text_input(
        "CBUAE Transfer Cutoff Date (YYYY-MM-DD)",
        value=st.session_state.get("cbuae_cutoff_date_str", DEFAULT_CBUAE_DATE),
        key="cbuae_cutoff_date_input"
    )

    # Store the threshold dates in session state
    try:
        cbuae_cutoff_date = datetime.strptime(cbuae_cutoff_str, "%Y-%m-%d")
        st.sidebar.caption(f"CBUAE cutoff: {cbuae_cutoff_date.strftime('%Y-%m-%d')}")

        # Store computed values in session state
        st.session_state["general_threshold_date"] = general_inactivity_threshold_date
        st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date
        st.session_state["cbuae_cutoff_date"] = cbuae_cutoff_date
        st.session_state["flagging_inactivity_threshold_days"] = general_inactivity_threshold_days
        st.session_state["freeze_inactivity_threshold_days"] = freeze_inactivity_threshold_days
        st.session_state["cbuae_cutoff_date_str"] = cbuae_cutoff_str

    except ValueError:
        st.sidebar.error("Invalid CBUAE cutoff date format. Using defaults.")
        # Store only the valid thresholds
        st.session_state["general_threshold_date"] = general_inactivity_threshold_date
        st.session_state["freeze_threshold_date"] = freeze_inactivity_threshold_date
        st.session_state["cbuae_cutoff_date"] = None
        st.session_state["flagging_inactivity_threshold_days"] = general_inactivity_threshold_days
        st.session_state["freeze_inactivity_threshold_days"] = freeze_inactivity_threshold_days


def show_database_status():
    """Show database connection status in sidebar."""
    st.sidebar.subheader("🔌 Database Status")
    
    if st.sidebar.button("Test Connection"):
        conn = get_db_connection()
        if conn:
            try:
                # Test with a simple query
                test_df = pd.read_sql("SELECT 1 as test", conn)
                if not test_df.empty:
                    st.sidebar.success("✅ Database Connected")
                else:
                    st.sidebar.error("❌ Database Query Failed")
            except Exception as e:
                st.sidebar.error(f"❌ Database Error: {e}")
        else:
            st.sidebar.error("❌ Database Connection Failed")


# Add database status to sidebar
def render_sidebar_with_status():
    """Enhanced sidebar with database status."""
    render_sidebar()
    
    # Add database status at the bottom
    st.sidebar.markdown("---")
    show_database_status()


# Helper function to safely get session state values
def safe_get_session_state(key, default=None):
    """Safely get a value from session state with a default."""
    return st.session_state.get(key, default)


# Initialize required session state variables
def initialize_session_state():
    """Initialize required session state variables."""
    if SESSION_COLUMN_MAPPING not in st.session_state:
        st.session_state[SESSION_COLUMN_MAPPING] = {}
    
    if SESSION_APP_DF not in st.session_state:
        st.session_state[SESSION_APP_DF] = None
        
    if SESSION_DATA_PROCESSED not in st.session_state:
        st.session_state[SESSION_DATA_PROCESSED] = False

    # Initialize analysis settings
    if "report_date_for_dormancy" not in st.session_state:
        st.session_state.report_date_for_dormancy = datetime.now().date()
        
    if "agent_name_for_compliance" not in st.session_state:
        st.session_state.agent_name_for_compliance = "SystemAuditor"


# Call initialization when module is imported
initialize_session_state()
