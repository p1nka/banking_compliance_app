import os
import sys
import io
import json
import secrets
import time
import traceback
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir))
sys.path.insert(0, project_root)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

# Import from the Banking Compliance app
from config import APP_TITLE, APP_SUBTITLE, DEFAULT_DORMANT_DAYS, DEFAULT_FREEZE_DAYS, DEFAULT_CBUAE_DATE
from data.parser import parse_data
from database.connection import get_db_connection
from database.operations import save_to_db, save_summary_to_db, log_flag_instructions
from agents.dormant import run_all_dormant_checks
from agents.compliance import run_all_compliance_checks, detect_incomplete_contact, detect_flag_candidates
from ai.llm import get_llm, get_fallback_response

# Fix for the Streamlit cache warning:
# This allows the app to run without a Streamlit runtime
os.environ["STREAMLIT_RUNTIME"] = "1"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
os.environ["STREAMLIT_SERVER_BASE_URL_PATH"] = "/"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


# Custom JSON encoder to handle NaN values, dates, and other problematic types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super().default(obj)


# Initialize the FastAPI app
app = FastAPI(
    title="Banking Compliance API",
    description="API endpoints for the Banking Compliance App",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBasic()


def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    """Validate API credentials."""
    # Get credentials from environment vars or use defaults
    correct_username = os.environ.get("APP_USERNAME", "admin")
    correct_password = os.environ.get("APP_PASSWORD", "pass123")

    is_username_correct = secrets.compare_digest(credentials.username, correct_username)
    is_password_correct = secrets.compare_digest(credentials.password, correct_password)

    if not (is_username_correct and is_password_correct):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username


# Global in-memory storage for data with TTL
DATA_STORE = {}
DATA_EXPIRY = {}  # Track when data should expire

# Data expiry time (4 hours)
DATA_TTL = 14400


@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "title": APP_TITLE,
        "description": APP_SUBTITLE,
        "version": "1.0.0",
        "endpoints": [
            "/upload-file",
            "/analyze/dormant",
            "/analyze/compliance",
            "/database/connect",
            "/database/query",
            "/flag-accounts"
        ]
    }


@app.get("/health")
def health_check():
    """Return the health status of the API."""
    try:
        # Check if database connection is working
        conn = get_db_connection()
        db_status = "connected" if conn else "disconnected"

        # Check if LLM is available
        llm = get_llm()
        ai_status = "available" if llm else "unavailable"

        # Clear expired data
        _clear_expired_data()

        return {
            "status": "healthy",
            "database": db_status,
            "ai_model": ai_status,
            "timestamp": datetime.now().isoformat(),
            "active_sessions": len(DATA_STORE)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _clear_expired_data():
    """Clear expired data from memory to prevent memory leaks."""
    current_time = time.time()
    expired_keys = [
        key for key, expiry_time in DATA_EXPIRY.items()
        if current_time > expiry_time
    ]

    for key in expired_keys:
        if key in DATA_STORE:
            del DATA_STORE[key]
        if key in DATA_EXPIRY:
            del DATA_EXPIRY[key]


@app.post("/upload-file")
async def upload_file(
        file: UploadFile = File(...),
        username: str = Depends(get_current_username)
):
    """
    Upload and parse a data file (CSV, Excel, JSON).
    The parsed data will be stored in memory for subsequent API calls.
    """
    try:
        # Read the file content
        content = await file.read()

        # Create file-like object
        if file.filename.endswith('.csv'):
            file_obj = io.StringIO(content.decode('utf-8'))
            df = pd.read_csv(file_obj)
        elif file.filename.endswith(('.xlsx', '.xls')):
            file_obj = io.BytesIO(content)
            df = pd.read_excel(file_obj)
        elif file.filename.endswith('.json'):
            file_obj = io.StringIO(content.decode('utf-8'))
            df = pd.read_json(file_obj)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV, Excel, or JSON.")

        # Parse the data using the app's parser
        parsed_df = parse_data(df)

        # Clean expired data before storing new data
        _clear_expired_data()

        # Store in memory with expiration
        session_id = secrets.token_hex(8)
        DATA_STORE[session_id] = parsed_df
        DATA_EXPIRY[session_id] = time.time() + DATA_TTL

        # Convert problematic values directly for JSON serialization
        preview_data = []
        for _, row in parsed_df.head(5).iterrows():
            record = {}
            for col in parsed_df.columns:
                val = row[col]
                if isinstance(val, (datetime, date, pd.Timestamp)):
                    record[col] = val.isoformat() if pd.notna(val) else None
                elif pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                    record[col] = None
                else:
                    record[col] = val
            preview_data.append(record)

        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded and parsed successfully",
            "session_id": session_id,
            "rows": len(parsed_df),
            "columns": list(parsed_df.columns),
            "preview": preview_data,
            "expires_in": f"{DATA_TTL / 3600:.1f} hours"
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/analyze/dormant")
async def analyze_dormant(
        session_id: str = Form(...),
        report_date: Optional[str] = Form(None),
        username: str = Depends(get_current_username)
):
    """
    Run dormant account analysis on previously uploaded data.
    """
    # Validate session and refresh expiry
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")

    # Refresh session expiry
    DATA_EXPIRY[session_id] = time.time() + DATA_TTL

    df = DATA_STORE[session_id]

    try:
        # Parse report date or use current date
        if report_date:
            report_date_obj = datetime.strptime(report_date, "%Y-%m-%d")
        else:
            report_date_obj = datetime.now()

        # Run the dormant account analysis
        results = run_all_dormant_checks(df, report_date_obj)

        # Clean statistics manually
        clean_stats = {}
        for key, value in results["statistics"].items():
            if isinstance(value, float) and np.isnan(value):
                clean_stats[key] = None
            elif pd.isna(value):
                clean_stats[key] = None
            elif isinstance(value, (datetime, date, pd.Timestamp)):
                clean_stats[key] = value.isoformat()
            else:
                clean_stats[key] = value

        # Create clean JSON-serializable result
        clean_results = {
            "total_accounts": results["total_accounts"],
            "statistics": clean_stats,
            "safe_deposit_count": results["sd"]["count"],
            "investment_count": results["inv"]["count"],
            "fixed_deposit_count": results["fd"]["count"],
            "demand_deposit_count": results["dd"]["count"],
            "bankers_cheques_count": results["chq"]["count"],
            "central_bank_transfer_count": results["cb"]["count"],
            "article3_process_count": results["art3"]["count"],
            "contact_attempts_count": results["con"]["count"]
        }

        return {
            "status": "success",
            "analysis_type": "dormant",
            "report_date": report_date_obj.strftime("%Y-%m-%d"),
            "results": clean_results
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error running dormant analysis: {str(e)}"
        )


@app.post("/analyze/compliance")
async def analyze_compliance(
        session_id: str = Form(...),
        username: str = Depends(get_current_username)
):
    """
    Run compliance analysis on previously uploaded data.
    """
    # Validate session and refresh expiry
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")

    # Refresh session expiry
    DATA_EXPIRY[session_id] = time.time() + DATA_TTL

    df = DATA_STORE[session_id]

    try:
        # Run the compliance analysis
        results = run_all_compliance_checks(df)

        # Convert to JSON-serializable format
        json_results = {
            "total_accounts": results["total_accounts"],
            "contact": {
                "count": results["contact"]["count"],
                "desc": results["contact"]["desc"]
            },
            "flag": {
                "count": results["flag"]["count"],
                "desc": results["flag"]["desc"]
            },
            "ledger": {
                "count": results["ledger"]["count"],
                "desc": results["ledger"]["desc"]
            },
            "freeze": {
                "count": results["freeze"]["count"],
                "desc": results["freeze"]["desc"]
            },
            "transfer": {
                "count": results["transfer"]["count"],
                "desc": results["transfer"]["desc"]
            },
            "foreign_currency": {
                "count": results["foreign_currency"]["count"],
                "desc": results["foreign_currency"]["desc"]
            },
            "safe_deposit": {
                "count": results["safe_deposit"]["count"],
                "desc": results["safe_deposit"]["desc"]
            },
            "payment_instruments": {
                "count": results["payment_instruments"]["count"],
                "desc": results["payment_instruments"]["desc"]
            },
            "claims": {
                "count": results["claims"]["count"],
                "desc": results["claims"]["desc"]
            }
        }

        return {
            "status": "success",
            "analysis_type": "compliance",
            "report_date": datetime.now().strftime("%Y-%m-%d"),
            "results": json_results
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error running compliance analysis: {str(e)}"
        )


@app.post("/database/connect")
async def connect_database(
        server: str = Form(...),
        database: str = Form(...),
        username: str = Form(...),
        password: str = Form(...),
        port: int = Form(1433),
        use_entra: bool = Form(False),
        entra_domain: Optional[str] = Form(None),
        api_user: str = Depends(get_current_username)
):
    """
    Test database connection with provided credentials.
    """
    try:
        # Set temporary environment variables for the connection
        os.environ["DB_SERVER_NAME"] = server
        os.environ["DB_NAME"] = database
        os.environ["DB_USERNAME"] = username
        os.environ["DB_PASSWORD"] = password
        os.environ["DB_PORT"] = str(port)

        if use_entra:
            if not entra_domain:
                raise HTTPException(status_code=400, detail="Entra domain is required when use_entra is True")
            os.environ["USE_ENTRA_AUTH"] = "true"
            os.environ["ENTRA_DOMAIN"] = entra_domain

        # Try to connect
        conn = get_db_connection()

        if conn:
            # Test the connection with a simple query
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()

            # Set session_id for database connection
            db_session_id = secrets.token_hex(8)

            return {
                "status": "success",
                "message": "Successfully connected to the database",
                "connection_id": db_session_id,
                "server": server,
                "database": database
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to establish database connection"
            )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Database connection error: {str(e)}"
        )
    finally:
        # Reset environment variables
        for key in ["DB_SERVER_NAME", "DB_NAME", "DB_USERNAME", "DB_PASSWORD",
                    "DB_PORT", "USE_ENTRA_AUTH", "ENTRA_DOMAIN"]:
            if key in os.environ:
                del os.environ[key]


@app.post("/database/query")
async def execute_query(
        query: str = Form(...),
        username: str = Depends(get_current_username)
):
    """
    Execute SQL query against the database.
    """
    try:
        # Get database connection
        conn = get_db_connection()

        if not conn:
            raise HTTPException(
                status_code=500,
                detail="Failed to establish database connection. Check your credentials in .streamlit/secrets.toml"
            )

        # Execute the query
        df = pd.read_sql(query, conn)

        # Clean results for JSON response
        # Convert DataFrame to dict manually to handle all problematic types
        records = []
        for _, row in df.iterrows():
            record = {}
            for col in df.columns:
                val = row[col]
                if isinstance(val, (datetime, date, pd.Timestamp)):
                    # Convert dates/timestamps to ISO format strings
                    record[col] = val.isoformat() if pd.notna(val) else None
                elif pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                    # Handle NaN and None
                    record[col] = None
                else:
                    # Regular values
                    record[col] = val
            records.append(record)

        return {
            "status": "success",
            "rows": len(df),
            "columns": list(df.columns),
            "results": records
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Query execution error: {str(e)}"
        )


@app.post("/flag-accounts")
async def flag_accounts(
        session_id: str = Form(...),
        account_ids: List[str] = Body(...),
        flag_instruction: str = Form("Dormant Account Flagged via API"),
        agent_name: str = Form("API Agent"),
        threshold_days: int = Form(DEFAULT_DORMANT_DAYS),
        username: str = Depends(get_current_username)
):
    """
    Flag accounts for dormancy review and log to database.
    """
    # Validate session and refresh expiry
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")

    # Refresh session expiry
    DATA_EXPIRY[session_id] = time.time() + DATA_TTL

    try:
        # Validate account_ids format
        if not account_ids:
            raise HTTPException(status_code=400, detail="No account IDs provided")

        # Handle case where account_ids might be a JSON string
        if isinstance(account_ids, str):
            try:
                # Try to parse as JSON
                account_ids = json.loads(account_ids)
            except json.JSONDecodeError:
                # If not JSON, treat as a single account ID
                account_ids = [account_ids]

        # Ensure account_ids is a list
        if not isinstance(account_ids, list):
            account_ids = [account_ids]

        # Convert all account IDs to strings
        account_ids = [str(account_id) for account_id in account_ids]

        # Get database connection to make sure it's available
        conn = get_db_connection()
        if not conn:
            raise HTTPException(
                status_code=500,
                detail="Failed to establish database connection. Check your database configuration."
            )

        # Log flag instructions to the database
        success, message = log_flag_instructions(
            account_ids=account_ids,
            flag_instruction=flag_instruction,
            flag_reason=f"Flagged by {agent_name} via API",
            flag_days=threshold_days,
            flagged_by=username
        )

        if success:
            return {
                "status": "success",
                "message": message,
                "flagged_accounts": account_ids,
                "threshold_days": threshold_days
            }
        else:
            raise HTTPException(status_code=500, detail=message)

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error flagging accounts: {str(e)}"
        )


@app.post("/analyze/ai-insights")
async def get_ai_insights(
        session_id: str = Form(...),
        analysis_type: str = Form(...),  # "dormant" or "compliance"
        insight_type: str = Form(...),  # "summary", "observation", "action", etc.
        username: str = Depends(get_current_username)
):
    """
    Generate AI insights based on previously run analysis.
    """
    # Validate session and refresh expiry
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")

    # Refresh session expiry
    DATA_EXPIRY[session_id] = time.time() + DATA_TTL

    try:
        # Get the LLM
        llm = get_llm()

        if not llm:
            # Use fallback response if LLM is not available
            fallback_text = get_fallback_response(insight_type)
            return {
                "status": "partial_success",
                "message": "AI model not available. Using fallback response.",
                "insight": fallback_text
            }

        # Sample data for insights
        df = DATA_STORE[session_id]
        sample_size = min(30, len(df))
        sample_df = df.sample(n=sample_size) if len(df) > 0 else df

        # Convert date columns to string to avoid serialization issues
        for col in sample_df.columns:
            if pd.api.types.is_datetime64_any_dtype(sample_df[col]):
                sample_df[col] = sample_df[col].astype(str)

        # Replace NaN with "NULL" string as placeholder
        sample_df_clean = sample_df.fillna("NULL")
        sample_data_csv = sample_df_clean.to_csv(index=False)

        # Generate insights based on the type
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from ai.llm import (
            DORMANT_SUMMARY_PROMPT,
            COMPLIANCE_SUMMARY_PROMPT,
            OBSERVATION_PROMPT,
            TREND_PROMPT,
            NARRATION_PROMPT,
            ACTION_PROMPT
        )

        output_parser = StrOutputParser()

        # Select prompt based on insight type and analysis type
        if analysis_type == "dormant":
            if insight_type == "summary":
                prompt = PromptTemplate.from_template(DORMANT_SUMMARY_PROMPT)
                chain = prompt | llm | output_parser
                insight = chain.invoke({"analysis_details": sample_data_csv})
            elif insight_type == "observation":
                prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                chain = prompt | llm | output_parser
                insight = chain.invoke({"data": sample_data_csv})
            elif insight_type == "trend":
                prompt = PromptTemplate.from_template(TREND_PROMPT)
                chain = prompt | llm | output_parser
                insight = chain.invoke({"data": sample_data_csv})
            elif insight_type == "action":
                # For action, we need observations and trends first
                obs_prompt = PromptTemplate.from_template(OBSERVATION_PROMPT)
                trend_prompt = PromptTemplate.from_template(TREND_PROMPT)

                obs_chain = obs_prompt | llm | output_parser
                trend_chain = trend_prompt | llm | output_parser

                observations = obs_chain.invoke({"data": sample_data_csv})
                trends = trend_chain.invoke({"data": sample_data_csv})

                action_prompt = PromptTemplate.from_template(ACTION_PROMPT)
                action_chain = action_prompt | llm | output_parser

                insight = action_chain.invoke({
                    "observation": observations,
                    "trend": trends
                })
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported insight type: {insight_type}")

        elif analysis_type == "compliance":
            if insight_type == "summary":
                prompt = PromptTemplate.from_template(COMPLIANCE_SUMMARY_PROMPT)
                chain = prompt | llm | output_parser
                insight = chain.invoke({"compliance_details": sample_data_csv})
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported insight type for compliance: {insight_type}")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported analysis type: {analysis_type}")

        # Save insights to database if appropriate
        if insight_type in ["observation", "trend", "action", "summary"]:
            try:
                observation = insight if insight_type == "observation" else "Generated via API"
                trend = insight if insight_type == "trend" else "Generated via API"
                narration = insight if insight_type == "summary" else "Generated via API"
                action = insight if insight_type == "action" else "Generated via API"

                save_summary_to_db(observation, trend, narration, action)
            except Exception as db_e:
                print(f"Warning: Could not save insight to database: {db_e}")

        return {
            "status": "success",
            "analysis_type": analysis_type,
            "insight_type": insight_type,
            "insight": insight
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating AI insights: {str(e)}"
        )


@app.post("/save-to-database")
async def save_data_to_db(
        session_id: str = Form(...),
        table_name: str = Form("accounts_data"),
        username: str = Depends(get_current_username)
):
    """
    Save the processed data to the database.
    """
    # Validate session and refresh expiry
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")

    # Refresh session expiry
    DATA_EXPIRY[session_id] = time.time() + DATA_TTL

    df = DATA_STORE[session_id]

    try:
        # Save to database
        success = save_to_db(df, table_name)

        if success:
            return {
                "status": "success",
                "message": f"Successfully saved {len(df)} rows to table '{table_name}'",
                "rows": len(df)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save data to table '{table_name}'"
            )

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error saving to database: {str(e)}"
        )


@app.get("/get-data-sample/{session_id}")
def get_data_sample(
        session_id: str,
        rows: int = Query(5, ge=1, le=100),
        username: str = Depends(get_current_username)
):
    """Get a sample of the data from a session."""
    # Validate session and refresh expiry
    if session_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail="Session not found. Please upload data first.")

    # Refresh session expiry
    DATA_EXPIRY[session_id] = time.time() + DATA_TTL

    df = DATA_STORE[session_id]

    # Clean sample data for JSON response
    sample_data = []
    for _, row in df.head(rows).iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if isinstance(val, (datetime, date, pd.Timestamp)):
                record[col] = val.isoformat() if pd.notna(val) else None
            elif pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                record[col] = None
            else:
                record[col] = val
        sample_data.append(record)

    return {
        "session_id": session_id,
        "total_rows": len(df),
        "columns": list(df.columns),
        "sample": sample_data,
        "expires_in": f"{(DATA_EXPIRY[session_id] - time.time()) / 3600:.1f} hours"
    }


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    # Set Streamlit environment variables to prevent warnings
    os.environ["STREAMLIT_RUNTIME"] = "1"
    os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "true"
    os.environ["STREAMLIT_SERVER_BASE_URL_PATH"] = "/"
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

    # Other startup tasks can be added here
    print(f"Banking Compliance API starting up at {datetime.now().isoformat()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run cleanup tasks on shutdown."""
    # Clear all data from memory
    DATA_STORE.clear()
    DATA_EXPIRY.clear()
    print(f"Banking Compliance API shutting down at {datetime.now().isoformat()}")


if __name__ == "__main__":  # Note: Fixed from "_main_"
    import uvicorn

    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)