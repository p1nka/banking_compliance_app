Banking Compliance API Documentation
This API provides RESTful endpoints to access the functionality of the Banking Compliance application, allowing you to:

Upload and process account data files
Run dormant account analysis per UAE regulations
Perform compliance checks against banking regulations
Generate AI-powered insights on dormant accounts and compliance issues
Interact with the database for account flagging and querying

API Endpoints
Authentication
All endpoints (except / and /health) require HTTP Basic authentication:

Username: admin (or as configured in environment)
Password: pass123 (or as configured in environment)

Core Endpoints
GET /
Returns information about the API and available endpoints.
GET /health
Checks the health and connectivity status of the API, database, and AI model.
POST /upload-file
Upload and process banking account data files. Supports CSV, Excel, and JSON formats.
Parameters:

file: The file to upload (multipart/form-data)

Returns:

session_id: Unique identifier for the uploaded data (used in subsequent calls)
rows: Number of rows processed
columns: List of column names
preview: Sample of the processed data

POST /analyze/dormant
Run dormant account analysis on previously uploaded data according to UAE banking regulations.
Parameters:

session_id: Session ID from previous file upload
report_date: (Optional) Reference date for analysis, defaults to current date

Returns:

Comprehensive analysis of dormant accounts by category
Statistical breakdown of dormant accounts

POST /analyze/compliance
Run compliance analysis on previously uploaded data to check for regulatory issues.
Parameters:

session_id: Session ID from previous file upload

Returns:

Detailed compliance issues by category
Counts of accounts requiring different types of compliance actions

Database Operations
POST /database/connect
Test database connection with provided credentials.
Parameters:

server: Database server name/address
database: Database name
username: Database username
password: Database password
port: Database port (default: 1433)
use_entra: Whether to use Microsoft Entra authentication (default: false)
entra_domain: Entra domain (required if use_entra is true)

Returns:

connection_id: Unique identifier for the database connection
Connection status and details

POST /database/query
Execute SQL query against the database.
Parameters:

query: SQL query string to execute

Returns:

rows: Number of rows returned
columns: List of column names
results: Query results as JSON

POST /save-to-database
Save processed data to a database table.
Parameters:

session_id: Session ID from previous file upload
table_name: Target table name (default: "accounts_data")

Returns:

Confirmation and row count of saved data

Account Operations
POST /flag-accounts
Flag accounts for dormancy review and log to the database.
Parameters:

session_id: Session ID from previous file upload
account_ids: List of account IDs to flag (JSON array in request body)
flag_instruction: Flag instruction text
agent_name: Name of the agent/system flagging the accounts
threshold_days: Threshold days for dormancy (default: 1095, or 3 years)

Returns:

Confirmation of flagged accounts

AI Insights
POST /analyze/ai-insights
Generate AI-powered insights based on previously run analysis.
Parameters:

session_id: Session ID from previous file upload
analysis_type: Type of analysis ("dormant" or "compliance")
insight_type: Type of insight to generate ("summary", "observation", "trend", "action")

Returns:

AI-generated insights based on the analysis

Data Access
GET /get-data-sample/{session_id}
Get a sample of the data from a session.
Parameters:

session_id: Session ID from previous file upload
rows: Number of rows to return (default: 5, max: 100)

Returns:

Data sample and metadata

Using with Postman

Authentication Setup:

Set up HTTP Basic authentication in Postman with the username and password


File Upload:

Use the /upload-file endpoint
Select "form-data" in the Body tab
Add a key "file" of type "File"
Select your CSV, Excel, or JSON file
Send the request and note the session_id in the response


Run Analysis:

Use the /analyze/dormant or /analyze/compliance endpoints
Select "form-data" in the Body tab
Add key "session_id" with the value from the upload response
Send the request


Generate Insights:

Use the /analyze/ai-insights endpoint
Add keys for "session_id", "analysis_type", and "insight_type"
Send the request


Database Operations:

Use the appropriate endpoints for database connectivity and queries
Follow the parameter requirements in the documentation



Error Handling
All endpoints return appropriate HTTP status codes:

200: Success
400: Bad request or invalid parameters
401: Unauthorized (authentication failed)
404: Resource not found
500: Server error

Detailed error messages are included in the response body.
Testing the API

Start the API server:

On Linux/Mac: ./run_api.sh
On Windows: run_api.bat


Run the test script:
python test_api.py path/to/your/data.csv

The test script will run through all endpoints and verify functionality.
