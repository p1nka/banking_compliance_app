from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import io
import json
import os
import tempfile
from pathlib import Path

# Import your existing modules
from agents.dormant import (
    run_all_dormant_identification_checks,
    check_safe_deposit_dormancy,
    check_investment_inactivity,
    check_fixed_deposit_inactivity,
    check_demand_deposit_inactivity,
    check_unclaimed_payment_instruments,
    check_eligible_for_cb_transfer,
    check_art3_process_needed,
    check_contact_attempts_needed,
    check_high_value_dormant_accounts,
    check_dormant_to_active_transitions
)

from agents.compliance import (
    run_all_compliance_checks,
    detect_incomplete_contact_attempts,
    detect_flag_candidates,
    detect_ledger_candidates,
    detect_freeze_candidates,
    detect_transfer_candidates_to_cb,
    detect_foreign_currency_conversion_needed,
    detect_sdb_court_application_needed,
    detect_unclaimed_payment_instruments_ledger,
    detect_claim_processing_pending,
    generate_annual_cbuae_report_summary,
    check_record_retention_compliance,
    log_flag_instructions
)

from agents.connector import (
    process_dormant_results_for_compliance,
    apply_dormant_flags_to_dataframe,
    run_dormant_then_compliance
)

from data.parser import parse_data
from database.connection import get_db_connection, ping_connections
from database.schema import init_db, get_db_schema
from database.operations import (
    save_to_db,
    save_summary_to_db,
    save_sql_query_to_history,
    get_recent_sql_history
)

# Initialize FastAPI app
app = FastAPI(
    title="UAE Banking Compliance Analysis API",
    description="API for dormant account identification and compliance analysis according to CBUAE regulations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class AccountData(BaseModel):
    Account_ID: str
    Customer_ID: str
    Account_Type: str
    Date_Last_Cust_Initiated_Activity: str
    Expected_Account_Dormant: str = "No"
    Current_Balance: Optional[float] = 0.0
    Currency: Optional[str] = "AED"


class DormantAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    report_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    dormant_flags_history: Optional[List[Dict[str, Any]]] = []


class ComplianceAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    general_threshold_days: int = 1095  # 3 years
    freeze_threshold_days: int = 1095  # 3 years
    agent_name: str = "APIAgent"


class SQLQueryRequest(BaseModel):
    query: str
    natural_language_query: Optional[str] = None


class FlagInstructionRequest(BaseModel):
    account_ids: List[str]
    agent_name: str
    threshold_days: Optional[int] = None


class AnalysisResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    count: Optional[int] = None
    description: Optional[str] = None


# Utility functions
def convert_dataframe_to_dict(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert DataFrame to JSON-serializable dictionary"""
    if df.empty:
        return {"data": [], "columns": [], "count": 0}

    return {
        "data": df.to_dict(orient='records'),
        "columns": df.columns.tolist(),
        "count": len(df)
    }


def dict_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of dictionaries to DataFrame"""
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


# Health check endpoint
@app.post("/connector/apply-dormant-flags")
async def apply_dormant_flags_endpoint(
        dormant_agent_results: Dict[str, Any],
        data: List[Dict[str, Any]]
):
    """Apply dormant flags to DataFrame"""
    try:
        df = dict_to_dataframe(data)
        modified_df = apply_dormant_flags_to_dataframe(df, dormant_agent_results)

        return {
            "success": True,
            "message": "Dormant flags applied successfully",
            "data": convert_dataframe_to_dict(modified_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flag application error: {str(e)}")


@app.post("/connector/run-dormant-then-compliance")
async def run_dormant_then_compliance_endpoint(
        data: List[Dict[str, Any]],
        report_date: str,
        flags_history: Optional[List[Dict[str, Any]]] = None,
        agent_name: str = "APIAgent"
):
    """Run dormant identification followed by compliance checks"""
    try:
        df = dict_to_dataframe(data)
        flags_history_df = dict_to_dataframe(flags_history) if flags_history else None

        dormant_results, compliance_results, enhanced_df = run_dormant_then_compliance(
            df, report_date, flags_history_df, agent_name
        )

        # Serialize results
        def serialize_results(results):
            serialized = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serialized_value = {}
                    for k, v in value.items():
                        if isinstance(v, pd.DataFrame):
                            serialized_value[k] = convert_dataframe_to_dict(v)
                        else:
                            serialized_value[k] = v
                    serialized[key] = serialized_value
                else:
                    serialized[key] = value
            return serialized

        return {
            "success": True,
            "message": "Dormant and compliance analysis completed",
            "dormant_results": serialize_results(dormant_results),
            "compliance_results": serialize_results(compliance_results),
            "enhanced_data": convert_dataframe_to_dict(enhanced_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Combined analysis error: {str(e)}")


# SQL and database operation endpoints
@app.post("/sql/execute")
async def execute_sql_query(request: SQLQueryRequest):
    """Execute SQL query on database"""
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Execute query
        result_df = pd.read_sql(request.query, conn)

        # Save to history if natural language query provided
        if request.natural_language_query:
            save_sql_query_to_history(request.natural_language_query, request.query)

        conn.close()

        return {
            "success": True,
            "message": f"Query executed successfully, returned {len(result_df)} rows",
            "data": convert_dataframe_to_dict(result_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL execution error: {str(e)}")


@app.get("/sql/history")
async def get_sql_query_history(limit: int = Query(default=10)):
    """Get recent SQL query history"""
    try:
        history_df = get_recent_sql_history(limit)

        if history_df is not None:
            return {
                "success": True,
                "message": f"Retrieved {len(history_df)} query history records",
                "data": convert_dataframe_to_dict(history_df)
            }
        else:
            return {
                "success": False,
                "message": "Failed to retrieve query history",
                "data": {"data": [], "columns": [], "count": 0}
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query history error: {str(e)}")


@app.post("/sql/save-summary")
async def save_analysis_summary(
        observation: str,
        trend: str,
        insight: str,
        action: str
):
    """Save analysis summary to database"""
    try:
        result = save_summary_to_db(observation, trend, insight, action)

        if result:
            return {"success": True, "message": "Summary saved successfully"}
        else:
            return {"success": False, "message": "Failed to save summary"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary save error: {str(e)}")


# File export endpoints
@app.get("/export/sample-data")
async def get_sample_data():
    """Get sample data for testing"""
    sample_data = [
        {
            "Account_ID": "ACC001",
            "Customer_ID": "CUST001",
            "Account_Type": "Savings",
            "Currency": "AED",
            "Current_Balance": 15000.50,
            "Date_Last_Cust_Initiated_Activity": "2021-03-15",
            "Expected_Account_Dormant": "Yes",
            "Customer_Address_Known": "No",
            "Customer_Has_Active_Liability_Account": "No"
        },
        {
            "Account_ID": "ACC002",
            "Customer_ID": "CUST002",
            "Account_Type": "Current",
            "Currency": "USD",
            "Current_Balance": 5000.00,
            "Date_Last_Cust_Initiated_Activity": "2023-01-20",
            "Expected_Account_Dormant": "No",
            "Customer_Address_Known": "Yes",
            "Customer_Has_Active_Liability_Account": "Yes"
        },
        {
            "Account_ID": "ACC003",
            "Customer_ID": "CUST003",
            "Account_Type": "Fixed Deposit",
            "Currency": "AED",
            "Current_Balance": 50000.00,
            "Date_Last_Cust_Initiated_Activity": "2020-12-01",
            "Expected_Account_Dormant": "Yes",
            "FTD_Maturity_Date": "2020-12-01",
            "FTD_Auto_Renewal": "No"
        },
        {
            "Account_ID": "ACC004",
            "Customer_ID": "CUST004",
            "Account_Type": "Safe Deposit Box",
            "Currency": "AED",
            "Current_Balance": 0.00,
            "Date_Last_Cust_Initiated_Activity": "2019-06-15",
            "Expected_Account_Dormant": "Yes",
            "SDB_Charges_Outstanding": 500.00,
            "Date_SDB_Charges_Became_Outstanding": "2021-06-15",
            "SDB_Tenant_Communication_Received": "No"
        },
        {
            "Account_ID": "ACC005",
            "Customer_ID": "CUST005",
            "Account_Type": "Investment",
            "Currency": "EUR",
            "Current_Balance": 25000.00,
            "Date_Last_Cust_Initiated_Activity": "2021-09-10",
            "Expected_Account_Dormant": "Yes",
            "Inv_Maturity_Redemption_Date": "2021-09-10",
            "Date_Last_Customer_Communication_Any_Type": "2021-08-01"
        }
    ]

    return {
        "success": True,
        "message": "Sample data retrieved",
        "data": sample_data,
        "count": len(sample_data)
    }


@app.post("/export/csv")
async def export_data_as_csv(data: List[Dict[str, Any]], filename: str = "export_data.csv"):
    """Export data as CSV file"""
    try:
        df = dict_to_dataframe(data)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            temp_path = tmp_file.name

        return FileResponse(
            path=temp_path,
            filename=filename,
            media_type='text/csv'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV export error: {str(e)}")


# Utility endpoints
@app.get("/utils/validate-data")
async def validate_data_structure(data: List[Dict[str, Any]]):
    """Validate data structure for compliance with system requirements"""
    try:
        df = dict_to_dataframe(data)

        required_columns = [
            "Account_ID", "Customer_ID", "Account_Type",
            "Date_Last_Cust_Initiated_Activity", "Expected_Account_Dormant"
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in required_columns]

        validation_result = {
            "valid": len(missing_columns) == 0,
            "row_count": len(df),
            "column_count": len(df.columns),
            "required_columns": required_columns,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "data_types": df.dtypes.to_dict()
        }

        return {
            "success": True,
            "message": "Data validation completed",
            "validation": validation_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data validation error: {str(e)}")


@app.get("/utils/column-mapping")
async def get_column_mapping():
    """Get standard column mapping for data import"""
    return {
        "success": True,
        "message": "Column mapping retrieved",
        "mapping": {
            "required_columns": {
                "Account_ID": "Unique account identifier",
                "Customer_ID": "Customer identifier",
                "Account_Type": "Type of account (Savings, Current, Fixed Deposit, etc.)",
                "Date_Last_Cust_Initiated_Activity": "Date of last customer-initiated activity (YYYY-MM-DD)",
                "Expected_Account_Dormant": "Whether account is expected to be dormant (Yes/No)"
            },
            "optional_columns": {
                "Current_Balance": "Account balance (numeric)",
                "Currency": "Account currency (AED, USD, EUR, etc.)",
                "Customer_Address_Known": "Whether customer address is known (Yes/No)",
                "Customer_Has_Active_Liability_Account": "Whether customer has active liability account (Yes/No)",
                "FTD_Maturity_Date": "Fixed deposit maturity date (YYYY-MM-DD)",
                "FTD_Auto_Renewal": "Fixed deposit auto renewal flag (Yes/No)",
                "SDB_Charges_Outstanding": "Safe deposit box outstanding charges (numeric)",
                "Date_SDB_Charges_Became_Outstanding": "Date SDB charges became outstanding (YYYY-MM-DD)",
                "SDB_Tenant_Communication_Received": "SDB tenant communication received (Yes/No)",
                "Inv_Maturity_Redemption_Date": "Investment maturity/redemption date (YYYY-MM-DD)"
            }
        }
    }


# System information endpoints
@app.get("/system/info")
async def get_system_info():
    """Get system information and capabilities"""
    return {
        "api_version": "1.0.0",
        "service_name": "UAE Banking Compliance Analysis API",
        "capabilities": {
            "dormant_analysis": [
                "Safe Deposit Box dormancy",
                "Investment account inactivity",
                "Fixed deposit inactivity",
                "Demand deposit inactivity",
                "Unclaimed payment instruments",
                "Central Bank transfer eligibility",
                "Article 3 process requirements",
                "Contact attempt requirements",
                "High-value dormant accounts",
                "Dormant-to-active transitions"
            ],
            "compliance_analysis": [
                "Incomplete contact attempts",
                "Flag candidates identification",
                "Internal ledger candidates",
                "Statement freeze requirements",
                "CBUAE transfer candidates",
                "Foreign currency conversion",
                "SDB court applications",
                "Unclaimed instruments ledger",
                "Claims processing monitoring",
                "Annual report generation",
                "Record retention compliance"
            ],
            "data_operations": [
                "Data parsing and standardization",
                "Database connectivity",
                "SQL query execution",
                "Data export capabilities",
                "Validation utilities"
            ]
        },
        "supported_formats": ["CSV", "XLSX", "JSON"],
        "regulations": ["CBUAE Dormant Accounts Regulation"],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/system/endpoints")
async def list_all_endpoints():
    """List all available API endpoints"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })

    return {
        "success": True,
        "message": "API endpoints retrieved",
        "endpoints": routes,
        "total_endpoints": len(routes)
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "message": "Endpoint not found",
            "detail": f"The requested endpoint {request.url.path} does not exist"
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "detail": "An unexpected error occurred while processing your request"
        }
    )


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    print("🏦 UAE Banking Compliance API starting up...")
    print("📊 Initializing database connections...")

    try:
        # Test database connection
        conn = get_db_connection()
        if conn:
            conn.close()
            print("✅ Database connection successful")
        else:
            print("⚠️ Database connection failed - some features may be limited")
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")

    print("🚀 API server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    print("🛑 UAE Banking Compliance API shutting down...")
    print("✅ Cleanup completed")


if __name__ == "__main__":
    import uvicorn

    # Run the application
    uvicorn.run(
        "banking_api:app",  # Updated to match the new filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ).get("/health")


async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "Banking Compliance API"
    }


# Database endpoints
@app.get("/database/status")
async def database_status():
    """Check database connection status"""
    try:
        conn = get_db_connection()
        if conn:
            conn.close()
            return {"status": "connected", "message": "Database connection successful"}
        else:
            return {"status": "disconnected", "message": "Database connection failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/database/initialize")
async def initialize_database():
    """Initialize database schema"""
    try:
        result = init_db()
        if result:
            return {"success": True, "message": "Database initialized successfully"}
        else:
            return {"success": False, "message": "Database initialization failed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization error: {str(e)}")


@app.get("/database/schema")
async def get_database_schema():
    """Get database schema information"""
    try:
        schema = get_db_schema()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema retrieval error: {str(e)}")


@app.post("/database/ping")
async def ping_database_connections():
    """Ping database connections"""
    try:
        ping_connections()
        return {"success": True, "message": "Database connections pinged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database ping error: {str(e)}")


# Data processing endpoints
@app.post("/data/parse")
async def parse_uploaded_data(file: UploadFile = File(...)):
    """Parse uploaded data file (CSV, XLSX, JSON)"""
    try:
        # Read uploaded file
        content = await file.read()

        # Determine file type and parse accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        elif file.filename.endswith('.json'):
            df = pd.read_json(io.StringIO(content.decode('utf-8')))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Parse data using your existing parser
        parsed_df = parse_data(df)

        return {
            "success": True,
            "message": f"Successfully parsed {len(parsed_df)} rows",
            "data": convert_dataframe_to_dict(parsed_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data parsing error: {str(e)}")


@app.post("/data/save")
async def save_data_to_database(data: List[Dict[str, Any]], table_name: str = "accounts_data"):
    """Save data to database"""
    try:
        df = dict_to_dataframe(data)
        result = save_to_db(df, table_name)

        if result:
            return {"success": True, "message": f"Successfully saved {len(df)} rows to {table_name}"}
        else:
            return {"success": False, "message": "Failed to save data to database"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data save error: {str(e)}")


# Dormant account analysis endpoints
@app.post("/dormant/analyze/all", response_model=Dict[str, Any])
async def run_all_dormant_analysis(request: DormantAnalysisRequest):
    """Run all dormant identification checks"""
    try:
        df = dict_to_dataframe(request.data)
        dormant_flags_history_df = dict_to_dataframe(
            request.dormant_flags_history) if request.dormant_flags_history else pd.DataFrame()

        results = run_all_dormant_identification_checks(
            df,
            report_date_str=request.report_date,
            dormant_flags_history_df=dormant_flags_history_df
        )

        # Convert DataFrames in results to serializable format
        serialized_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serialized_value = {}
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        serialized_value[k] = convert_dataframe_to_dict(v)
                    else:
                        serialized_value[k] = v
                serialized_results[key] = serialized_value
            else:
                serialized_results[key] = value

        return {
            "success": True,
            "message": "Dormant analysis completed successfully",
            "results": serialized_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dormant analysis error: {str(e)}")


@app.post("/dormant/sdb")
async def analyze_safe_deposit_dormancy(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze Safe Deposit Box dormancy"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_safe_deposit_dormancy(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="SDB dormancy analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDB dormancy analysis error: {str(e)}")


@app.post("/dormant/investment")
async def analyze_investment_inactivity(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze Investment account inactivity"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_investment_inactivity(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="Investment inactivity analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investment inactivity analysis error: {str(e)}")


@app.post("/dormant/fixed-deposit")
async def analyze_fixed_deposit_inactivity(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze Fixed Deposit inactivity"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_fixed_deposit_inactivity(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="Fixed deposit inactivity analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fixed deposit inactivity analysis error: {str(e)}")


@app.post("/dormant/demand-deposit")
async def analyze_demand_deposit_inactivity(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze Demand Deposit inactivity"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_demand_deposit_inactivity(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="Demand deposit inactivity analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demand deposit inactivity analysis error: {str(e)}")


@app.post("/dormant/unclaimed-instruments")
async def analyze_unclaimed_payment_instruments(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze unclaimed payment instruments"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_unclaimed_payment_instruments(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="Unclaimed payment instruments analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unclaimed instruments analysis error: {str(e)}")


@app.post("/dormant/cb-transfer-eligible")
async def analyze_cb_transfer_eligibility(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze accounts eligible for Central Bank transfer"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_eligible_for_cb_transfer(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="CB transfer eligibility analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CB transfer eligibility analysis error: {str(e)}")


@app.post("/dormant/art3-process")
async def analyze_art3_process_needed(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze accounts needing Article 3 process"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_art3_process_needed(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="Article 3 process analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Article 3 process analysis error: {str(e)}")


@app.post("/dormant/contact-attempts")
async def analyze_contact_attempts_needed(data: List[Dict[str, Any]], report_date: str = Query(default=None)):
    """Analyze accounts needing proactive contact attempts"""
    try:
        df = dict_to_dataframe(data)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_contact_attempts_needed(df, report_date_obj)

        return AnalysisResponse(
            success=True,
            message="Contact attempts analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contact attempts analysis error: {str(e)}")


@app.post("/dormant/high-value")
async def analyze_high_value_dormant(data: List[Dict[str, Any]], threshold_balance: float = Query(default=25000)):
    """Analyze high-value dormant accounts"""
    try:
        df = dict_to_dataframe(data)

        result_df, count, description, details = check_high_value_dormant_accounts(df, threshold_balance)

        return AnalysisResponse(
            success=True,
            message="High-value dormant analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"High-value dormant analysis error: {str(e)}")


@app.post("/dormant/transitions")
async def analyze_dormant_to_active_transitions(
        data: List[Dict[str, Any]],
        dormant_flags_history: List[Dict[str, Any]] = [],
        report_date: str = Query(default=None),
        activity_lookback_days: int = Query(default=30)
):
    """Analyze dormant-to-active transitions"""
    try:
        df = dict_to_dataframe(data)
        dormant_flags_history_df = dict_to_dataframe(dormant_flags_history)
        report_date_obj = datetime.strptime(report_date or datetime.now().strftime("%Y-%m-%d"), "%Y-%m-%d")

        result_df, count, description, details = check_dormant_to_active_transitions(
            df, report_date_obj, dormant_flags_history_df, activity_lookback_days
        )

        return AnalysisResponse(
            success=True,
            message="Dormant-to-active transitions analysis completed",
            data={
                "results": convert_dataframe_to_dict(result_df),
                "details": details
            },
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dormant-to-active transitions analysis error: {str(e)}")


# Compliance analysis endpoints
@app.post("/compliance/analyze/all")
async def run_all_compliance_analysis(request: ComplianceAnalysisRequest):
    """Run all compliance checks"""
    try:
        df = dict_to_dataframe(request.data)
        general_threshold_date = datetime.now() - timedelta(days=request.general_threshold_days)
        freeze_threshold_date = datetime.now() - timedelta(days=request.freeze_threshold_days)

        results = run_all_compliance_checks(
            df,
            general_threshold_date=general_threshold_date,
            freeze_threshold_date=freeze_threshold_date,
            agent_name=request.agent_name
        )

        # Convert DataFrames in results to serializable format
        serialized_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serialized_value = {}
                for k, v in value.items():
                    if isinstance(v, pd.DataFrame):
                        serialized_value[k] = convert_dataframe_to_dict(v)
                    else:
                        serialized_value[k] = v
                serialized_results[key] = serialized_value
            else:
                serialized_results[key] = value

        return {
            "success": True,
            "message": "Compliance analysis completed successfully",
            "results": serialized_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compliance analysis error: {str(e)}")


@app.post("/compliance/incomplete-contact")
async def analyze_incomplete_contact_attempts(data: List[Dict[str, Any]]):
    """Analyze incomplete contact attempts"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_incomplete_contact_attempts(df)

        return AnalysisResponse(
            success=True,
            message="Incomplete contact attempts analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Incomplete contact analysis error: {str(e)}")


@app.post("/compliance/flag-candidates")
async def analyze_flag_candidates(data: List[Dict[str, Any]], threshold_days: int = Query(default=1095)):
    """Analyze accounts that should be flagged as dormant"""
    try:
        df = dict_to_dataframe(data)
        inactivity_threshold_date = datetime.now() - timedelta(days=threshold_days)

        result_df, count, description = detect_flag_candidates(df, inactivity_threshold_date)

        return AnalysisResponse(
            success=True,
            message="Flag candidates analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flag candidates analysis error: {str(e)}")


@app.post("/compliance/ledger-candidates")
async def analyze_ledger_candidates(data: List[Dict[str, Any]]):
    """Analyze accounts for internal ledger transfer"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_ledger_candidates(df)

        return AnalysisResponse(
            success=True,
            message="Ledger candidates analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ledger candidates analysis error: {str(e)}")


@app.post("/compliance/statement-freeze")
async def analyze_statement_freeze_candidates(data: List[Dict[str, Any]], threshold_days: int = Query(default=1095)):
    """Analyze accounts needing statement freeze"""
    try:
        df = dict_to_dataframe(data)
        freeze_threshold_date = datetime.now() - timedelta(days=threshold_days)

        result_df, count, description = detect_freeze_candidates(df, freeze_threshold_date)

        return AnalysisResponse(
            success=True,
            message="Statement freeze candidates analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Statement freeze analysis error: {str(e)}")


@app.post("/compliance/cb-transfer-candidates")
async def analyze_cb_transfer_candidates(data: List[Dict[str, Any]]):
    """Analyze accounts for CBUAE transfer"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_transfer_candidates_to_cb(df)

        return AnalysisResponse(
            success=True,
            message="CBUAE transfer candidates analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CBUAE transfer candidates analysis error: {str(e)}")


@app.post("/compliance/fx-conversion")
async def analyze_fx_conversion_needed(data: List[Dict[str, Any]]):
    """Analyze foreign currency accounts needing conversion"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_foreign_currency_conversion_needed(df)

        return AnalysisResponse(
            success=True,
            message="Foreign currency conversion analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FX conversion analysis error: {str(e)}")


@app.post("/compliance/sdb-court-application")
async def analyze_sdb_court_application(data: List[Dict[str, Any]]):
    """Analyze SDBs needing court application"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_sdb_court_application_needed(df)

        return AnalysisResponse(
            success=True,
            message="SDB court application analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SDB court application analysis error: {str(e)}")


@app.post("/compliance/unclaimed-instruments-ledger")
async def analyze_unclaimed_instruments_ledger(data: List[Dict[str, Any]]):
    """Analyze unclaimed instruments for ledger"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_unclaimed_payment_instruments_ledger(df)

        return AnalysisResponse(
            success=True,
            message="Unclaimed instruments ledger analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unclaimed instruments ledger analysis error: {str(e)}")


@app.post("/compliance/claims-processing")
async def analyze_claims_processing_pending(data: List[Dict[str, Any]]):
    """Analyze pending claims processing"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = detect_claim_processing_pending(df)

        return AnalysisResponse(
            success=True,
            message="Claims processing analysis completed",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claims processing analysis error: {str(e)}")


@app.post("/compliance/annual-report")
async def generate_annual_report_summary(data: List[Dict[str, Any]]):
    """Generate annual CBUAE report summary"""
    try:
        df = dict_to_dataframe(data)
        result_df, count, description = generate_annual_cbuae_report_summary(df)

        return AnalysisResponse(
            success=True,
            message="Annual report summary generated",
            data={"results": convert_dataframe_to_dict(result_df)},
            count=count,
            description=description
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Annual report generation error: {str(e)}")


@app.post("/compliance/record-retention")
async def analyze_record_retention_compliance(data: List[Dict[str, Any]]):
    """Analyze record retention compliance"""
    try:
        df = dict_to_dataframe(data)
        not_compliant_df, compliant_df, description = check_record_retention_compliance(df)

        return {
            "success": True,
            "message": "Record retention compliance analysis completed",
            "data": {
                "not_compliant": convert_dataframe_to_dict(not_compliant_df),
                "compliant": convert_dataframe_to_dict(compliant_df)
            },
            "description": description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Record retention analysis error: {str(e)}")


@app.post("/compliance/log-flags")
async def log_flag_instructions_endpoint(request: FlagInstructionRequest):
    """Log flag instructions for accounts"""
    try:
        status, message = log_flag_instructions(
            request.account_ids,
            request.agent_name,
            request.threshold_days
        )

        return {
            "success": status,
            "message": message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flag logging error: {str(e)}")


# Connector endpoints
@app.post("/connector/process-dormant-for-compliance")
async def process_dormant_for_compliance(
        dormant_results: Dict[str, Any],
        data: List[Dict[str, Any]]
):
    """Process dormant results for compliance analysis"""
    try:
        df = dict_to_dataframe(data)
        enhanced_df = process_dormant_results_for_compliance(dormant_results, df)

        return {
            "success": True,
            "message": "Dormant results processed for compliance",
            "data": convert_dataframe_to_dict(enhanced_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dormant processing error: {str(e)}")

@app.get("/health/")
async def health_check_with_slash():
    """Health check endpoint with trailing slash"""
    return await health_check()
