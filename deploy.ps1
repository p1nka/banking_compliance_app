# Banking Compliance Application Deployment Script for Windows PowerShell
# This script helps deploy the application using Docker on Windows

param(
    [Parameter(Position=0)]
    [string]$Command = "help",

    [Parameter(Position=1)]
    [string]$Mode = "streamlit",

    [Parameter(Position=2)]
    [string]$Service = "",

    [Parameter(Position=3)]
    [int]$Lines = 50
)

# Configuration
$APP_NAME = "banking-compliance"
$DOCKER_COMPOSE_FILE = "docker-compose.yml"
$ENV_FILE = ".env"
$SECRETS_FILE = ".streamlit\secrets.toml"

# Functions for colored output
function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Header {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Blue
    Write-Host "  Banking Compliance Application Deployment" -ForegroundColor Blue
    Write-Host "==================================================" -ForegroundColor Blue
    Write-Host ""
}

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."

    # Check if Docker is installed
    try {
        $null = docker --version
        if ($LASTEXITCODE -ne 0) { throw }
    }
    catch {
        Write-Error "Docker is not installed. Please install Docker Desktop first."
        Write-Host "Download from: https://www.docker.com/products/docker-desktop"
        exit 1
    }

    # Check if Docker Compose is available
    try {
        $null = docker-compose --version
        if ($LASTEXITCODE -ne 0) {
            $null = docker compose version
            if ($LASTEXITCODE -ne 0) { throw }
        }
    }
    catch {
        Write-Error "Docker Compose is not available. Please install Docker Desktop with Compose."
        exit 1
    }

    # Check if Docker daemon is running
    try {
        $null = docker info 2>$null
        if ($LASTEXITCODE -ne 0) { throw }
    }
    catch {
        Write-Error "Docker daemon is not running. Please start Docker Desktop first."
        exit 1
    }

    Write-Success "All prerequisites met"
}

function Initialize-Environment {
    Write-Info "Setting up environment configuration..."

    # Create .env file if it doesn't exist
    if (-not (Test-Path $ENV_FILE)) {
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" $ENV_FILE
            Write-Warning "Created $ENV_FILE from .env.example. Please update it with your actual values."
        }
        else {
            Write-Error ".env.example not found. Cannot create environment file."
            exit 1
        }
    }
    else {
        Write-Success "Environment file already exists"
    }

    # Create .streamlit directory if it doesn't exist
    if (-not (Test-Path ".streamlit")) {
        New-Item -ItemType Directory -Path ".streamlit" | Out-Null
    }

    # Create secrets file if it doesn't exist
    if (-not (Test-Path $SECRETS_FILE)) {
        Write-Warning "Creating default secrets.toml file. Please update it with your actual credentials."

        $secretsContent = @"
# Banking Compliance App Secrets
# Update these values with your actual credentials

# GROQ API Key for AI features
GROQ_API_KEY = "your_groq_api_key_here"

# Application Authentication
APP_USERNAME = "admin"
APP_PASSWORD = "admin_password"

# Azure SQL Database Configuration
DB_SERVER_NAME = "your_server.database.windows.net"
DB_NAME = "compliance_db"
DB_USERNAME = "your_username"
DB_PASSWORD = "your_password"
DB_PORT = "1433"

# Optional: Microsoft Entra Authentication
USE_ENTRA_AUTH = "false"
ENTRA_DOMAIN = "yourdomain.onmicrosoft.com"
"@

        $secretsContent | Out-File -FilePath $SECRETS_FILE -Encoding UTF8
    }
    else {
        Write-Success "Secrets file already exists"
    }

    # Create logs directory
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" | Out-Null
    }

    Write-Success "Environment setup completed"
}

function Test-Configuration {
    Write-Info "Validating configuration..."

    # Check if required environment variables are set in .env
    if (Test-Path $ENV_FILE) {
        $envContent = Get-Content $ENV_FILE -Raw
        if ($envContent -like "*your_groq_api_key_here*") {
            Write-Warning "Please update GROQ_API_KEY in $ENV_FILE"
        }
        if ($envContent -like "*your_server.database.windows.net*") {
            Write-Warning "Please update database configuration in $ENV_FILE"
        }
    }

    # Check secrets file
    if (Test-Path $SECRETS_FILE) {
        $secretsContent = Get-Content $SECRETS_FILE -Raw
        if ($secretsContent -like "*your_groq_api_key_here*") {
            Write-Warning "Please update GROQ_API_KEY in $SECRETS_FILE"
        }
        if ($secretsContent -like "*your_server.database.windows.net*") {
            Write-Warning "Please update database configuration in $SECRETS_FILE"
        }
    }

    Write-Success "Configuration validation completed"
}

function Build-Images {
    Write-Info "Building Docker images..."

    try {
        docker-compose build
        if ($LASTEXITCODE -ne 0) {
            docker compose build
            if ($LASTEXITCODE -ne 0) { throw }
        }
    }
    catch {
        Write-Error "Failed to build Docker images"
        exit 1
    }

    Write-Success "Docker images built successfully"
}

function Deploy-Application {
    param([string]$DeployMode)

    Write-Info "Deploying application in $DeployMode mode..."

    switch ($DeployMode) {
        "streamlit" {
            try {
                docker-compose up -d banking-compliance-app redis
                if ($LASTEXITCODE -ne 0) {
                    docker compose up -d banking-compliance-app redis
                }
            }
            catch {
                Write-Error "Failed to deploy Streamlit application"
                exit 1
            }
            Write-Success "Streamlit application deployed"
            Write-Info "Streamlit app will be available at: http://localhost:8501"
        }
        "api" {
            try {
                docker-compose up -d banking-compliance-api redis
                if ($LASTEXITCODE -ne 0) {
                    docker compose up -d banking-compliance-api redis
                }
            }
            catch {
                Write-Error "Failed to deploy API application"
                exit 1
            }
            Write-Success "API application deployed"
            Write-Info "API will be available at: http://localhost:8000"
            Write-Info "API documentation: http://localhost:8000/docs"
        }
        "both" {
            try {
                docker-compose up -d banking-compliance-app banking-compliance-api redis
                if ($LASTEXITCODE -ne 0) {
                    docker compose up -d banking-compliance-app banking-compliance-api redis
                }
            }
            catch {
                Write-Error "Failed to deploy both applications"
                exit 1
            }
            Write-Success "Both applications deployed"
            Write-Info "Streamlit app: http://localhost:8501"
            Write-Info "API: http://localhost:8000"
            Write-Info "API documentation: http://localhost:8000/docs"
        }
        "full" {
            try {
                docker-compose --profile with-nginx up -d
                if ($LASTEXITCODE -ne 0) {
                    docker compose --profile with-nginx up -d
                }
            }
            catch {
                Write-Error "Failed to deploy full stack"
                exit 1
            }
            Write-Success "Full stack deployed with nginx"
            Write-Info "Application available at: http://localhost"
        }
        "dev" {
            try {
                docker-compose --profile with-postgres up -d
                if ($LASTEXITCODE -ne 0) {
                    docker compose --profile with-postgres up -d
                }
            }
            catch {
                Write-Error "Failed to deploy development environment"
                exit 1
            }
            Write-Success "Development environment deployed with PostgreSQL"
            Write-Info "Streamlit app: http://localhost:8501"
            Write-Info "API: http://localhost:8000"
            Write-Info "PostgreSQL: localhost:5432"
        }
        default {
            Write-Error "Unknown deployment mode: $DeployMode"
            Write-Info "Available modes: streamlit, api, both, full, dev"
            exit 1
        }
    }
}

function Show-Status {
    Write-Info "Checking application status..."

    try {
        docker-compose ps
        if ($LASTEXITCODE -ne 0) {
            docker compose ps
        }
    }
    catch {
        Write-Error "Failed to get container status"
    }

    Write-Host ""
    Write-Info "Container logs (last 10 lines):"

    # Show logs for running containers
    $containers = docker ps --format "{{.Names}}" | Where-Object { $_ -like "*banking-compliance*" }
    foreach ($container in $containers) {
        Write-Host "--- Logs for $container ---" -ForegroundColor Blue
        docker logs --tail 10 $container
        Write-Host ""
    }
}

function Stop-Application {
    Write-Info "Stopping application..."

    try {
        docker-compose down
        if ($LASTEXITCODE -ne 0) {
            docker compose down
        }
    }
    catch {
        Write-Error "Failed to stop application"
        exit 1
    }

    Write-Success "Application stopped"
}

function Remove-Resources {
    Write-Info "Cleaning up Docker resources..."

    # Stop and remove containers
    try {
        docker-compose down -v --remove-orphans
        if ($LASTEXITCODE -ne 0) {
            docker compose down -v --remove-orphans
        }
    }
    catch {
        Write-Warning "Some containers could not be removed"
    }

    # Remove unused images
    docker image prune -f

    # Remove unused volumes (with confirmation)
    $answer = Read-Host "Do you want to remove unused volumes? This will delete persistent data. (y/N)"
    if ($answer -eq "y" -or $answer -eq "Y") {
        docker volume prune -f
        Write-Warning "Unused volumes removed"
    }

    Write-Success "Cleanup completed"
}

function Show-Logs {
    param([string]$ServiceName, [int]$LogLines)

    if ([string]::IsNullOrEmpty($ServiceName)) {
        Write-Error "Please specify a service name"
        Write-Info "Available services: banking-compliance-app, banking-compliance-api, redis, nginx, postgres"
        exit 1
    }

    Write-Info "Showing last $LogLines lines of logs for $ServiceName..."

    try {
        docker-compose logs --tail $LogLines -f $ServiceName
        if ($LASTEXITCODE -ne 0) {
            docker compose logs --tail $LogLines -f $ServiceName
        }
    }
    catch {
        Write-Error "Failed to show logs for $ServiceName"
    }
}

function Test-Health {
    Write-Info "Performing health check..."

    $appHealthy = $false
    $apiHealthy = $false
    $redisHealthy = $false

    # Check Streamlit app
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "Streamlit app is healthy"
            $appHealthy = $true
        }
    }
    catch {
        Write-Error "Streamlit app is not responding"
    }

    # Check API
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Success "API is healthy"
            $apiHealthy = $true
        }
    }
    catch {
        Write-Error "API is not responding"
    }

    # Check Redis
    try {
        $result = docker exec banking-compliance-redis redis-cli ping 2>$null
        if ($result -eq "PONG") {
            Write-Success "Redis is healthy"
            $redisHealthy = $true
        }
    }
    catch {
        Write-Error "Redis is not responding"
    }

    if ($appHealthy -and $apiHealthy -and $redisHealthy) {
        Write-Success "All services are healthy"
        return $true
    }
    else {
        Write-Error "Some services are not healthy"
        return $false
    }
}

function Show-Help {
    Write-Host "Banking Compliance Application Deployment Script for Windows PowerShell"
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [COMMAND] [OPTIONS]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  deploy [MODE]     Deploy the application"
    Write-Host "                    Modes: streamlit, api, both, full, dev"
    Write-Host "  status           Show application status"
    Write-Host "  logs [SERVICE]   Show logs for a service"
    Write-Host "  stop             Stop the application"
    Write-Host "  restart [MODE]   Restart the application"
    Write-Host "  cleanup          Stop and remove all containers and images"
    Write-Host "  health           Perform health check on all services"
    Write-Host "  setup            Setup environment configuration"
    Write-Host "  help             Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\deploy.ps1 deploy streamlit    # Deploy only Streamlit app"
    Write-Host "  .\deploy.ps1 deploy both         # Deploy both Streamlit and API"
    Write-Host "  .\deploy.ps1 logs banking-compliance-app  # Show app logs"
    Write-Host "  .\deploy.ps1 status              # Show status of all services"
    Write-Host "  .\deploy.ps1 health              # Check health of all services"
    Write-Host ""
}

# Main script execution
Write-Header

switch ($Command.ToLower()) {
    "deploy" {
        Test-Prerequisites
        Initialize-Environment
        Test-Configuration
        Build-Images
        Deploy-Application $Mode
        Write-Host ""
        Show-Status
        Write-Host ""
        Test-Health
    }
    "status" {
        Show-Status
    }
    "logs" {
        if ([string]::IsNullOrEmpty($Service)) {
            $Service = "banking-compliance-app"
        }
        Show-Logs $Service $Lines
    }
    "stop" {
        Stop-Application
    }
    "restart" {
        Stop-Application
        Start-Sleep -Seconds 2
        Deploy-Application $Mode
    }
    "cleanup" {
        Remove-Resources
    }
    "health" {
        Test-Health
    }
    "setup" {
        Initialize-Environment
        Test-Configuration
    }
    "help" {
        Show-Help
    }
    default {
        Write-Error "Unknown command: $Command"
        Write-Host ""
        Show-Help
        exit 1
    }
}