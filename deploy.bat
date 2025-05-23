@echo off
REM Banking Compliance Application Deployment Script for Windows
REM This script helps deploy the application using Docker on Windows

setlocal enabledelayedexpansion

REM Configuration
set APP_NAME=banking-compliance
set DOCKER_COMPOSE_FILE=docker-compose.yml
set ENV_FILE=.env
set SECRETS_FILE=.streamlit\secrets.toml

REM Colors (limited in batch)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set NC=[0m

goto main

:print_header
echo.
echo ==================================================
echo   Banking Compliance Application Deployment
echo ==================================================
echo.
goto :eof

:print_success
echo %GREEN%✓ %~1%NC%
goto :eof

:print_warning
echo %YELLOW%⚠ %~1%NC%
goto :eof

:print_error
echo %RED%✗ %~1%NC%
goto :eof

:print_info
echo %BLUE%ℹ %~1%NC%
goto :eof

:check_prerequisites
call :print_info "Checking prerequisites..."

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not installed. Please install Docker Desktop first."
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    docker compose version >nul 2>&1
    if errorlevel 1 (
        call :print_error "Docker Compose is not available. Please install Docker Desktop with Compose."
        pause
        exit /b 1
    )
)

REM Check if Docker daemon is running
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker daemon is not running. Please start Docker Desktop first."
    pause
    exit /b 1
)

call :print_success "All prerequisites met"
goto :eof

:setup_environment
call :print_info "Setting up environment configuration..."

REM Create .env file if it doesn't exist
if not exist "%ENV_FILE%" (
    if exist ".env.example" (
        copy ".env.example" "%ENV_FILE%" >nul
        call :print_warning "Created %ENV_FILE% from .env.example. Please update it with your actual values."
    ) else (
        call :print_error ".env.example not found. Cannot create environment file."
        pause
        exit /b 1
    )
) else (
    call :print_success "Environment file already exists"
)

REM Create .streamlit directory if it doesn't exist
if not exist ".streamlit" mkdir ".streamlit"

REM Create secrets file if it doesn't exist
if not exist "%SECRETS_FILE%" (
    call :print_warning "Creating default secrets.toml file. Please update it with your actual credentials."
    (
        echo # Banking Compliance App Secrets
        echo # Update these values with your actual credentials
        echo.
        echo # GROQ API Key for AI features
        echo GROQ_API_KEY = "your_groq_api_key_here"
        echo.
        echo # Application Authentication
        echo APP_USERNAME = "admin"
        echo APP_PASSWORD = "admin_password"
        echo.
        echo # Azure SQL Database Configuration
        echo DB_SERVER_NAME = "your_server.database.windows.net"
        echo DB_NAME = "compliance_db"
        echo DB_USERNAME = "your_username"
        echo DB_PASSWORD = "your_password"
        echo DB_PORT = "1433"
        echo.
        echo # Optional: Microsoft Entra Authentication
        echo USE_ENTRA_AUTH = "false"
        echo ENTRA_DOMAIN = "yourdomain.onmicrosoft.com"
    ) > "%SECRETS_FILE%"
) else (
    call :print_success "Secrets file already exists"
)

REM Create logs directory
if not exist "logs" mkdir "logs"

call :print_success "Environment setup completed"
goto :eof

:validate_configuration
call :print_info "Validating configuration..."

REM Check if required environment variables are set in .env
findstr /C:"your_groq_api_key_here" "%ENV_FILE%" >nul 2>&1
if not errorlevel 1 (
    call :print_warning "Please update GROQ_API_KEY in %ENV_FILE%"
)

findstr /C:"your_server.database.windows.net" "%ENV_FILE%" >nul 2>&1
if not errorlevel 1 (
    call :print_warning "Please update database configuration in %ENV_FILE%"
)

REM Check secrets file
findstr /C:"your_groq_api_key_here" "%SECRETS_FILE%" >nul 2>&1
if not errorlevel 1 (
    call :print_warning "Please update GROQ_API_KEY in %SECRETS_FILE%"
)

findstr /C:"your_server.database.windows.net" "%SECRETS_FILE%" >nul 2>&1
if not errorlevel 1 (
    call :print_warning "Please update database configuration in %SECRETS_FILE%"
)

call :print_success "Configuration validation completed"
goto :eof

:build_images
call :print_info "Building Docker images..."

REM Try docker-compose first, then docker compose
docker-compose build
if errorlevel 1 (
    docker compose build
    if errorlevel 1 (
        call :print_error "Failed to build Docker images"
        pause
        exit /b 1
    )
)

call :print_success "Docker images built successfully"
goto :eof

:deploy_application
set mode=%~1
call :print_info "Deploying application in %mode% mode..."

if "%mode%"=="streamlit" (
    docker-compose up -d banking-compliance-app redis
    if errorlevel 1 docker compose up -d banking-compliance-app redis
    call :print_success "Streamlit application deployed"
    call :print_info "Streamlit app will be available at: http://localhost:8501"
) else if "%mode%"=="api" (
    docker-compose up -d banking-compliance-api redis
    if errorlevel 1 docker compose up -d banking-compliance-api redis
    call :print_success "API application deployed"
    call :print_info "API will be available at: http://localhost:8000"
    call :print_info "API documentation: http://localhost:8000/docs"
) else if "%mode%"=="both" (
    docker-compose up -d banking-compliance-app banking-compliance-api redis
    if errorlevel 1 docker compose up -d banking-compliance-app banking-compliance-api redis
    call :print_success "Both applications deployed"
    call :print_info "Streamlit app: http://localhost:8501"
    call :print_info "API: http://localhost:8000"
    call :print_info "API documentation: http://localhost:8000/docs"
) else if "%mode%"=="full" (
    docker-compose --profile with-nginx up -d
    if errorlevel 1 docker compose --profile with-nginx up -d
    call :print_success "Full stack deployed with nginx"
    call :print_info "Application available at: http://localhost"
) else if "%mode%"=="dev" (
    docker-compose --profile with-postgres up -d
    if errorlevel 1 docker compose --profile with-postgres up -d
    call :print_success "Development environment deployed with PostgreSQL"
    call :print_info "Streamlit app: http://localhost:8501"
    call :print_info "API: http://localhost:8000"
    call :print_info "PostgreSQL: localhost:5432"
) else (
    call :print_error "Unknown deployment mode: %mode%"
    call :print_info "Available modes: streamlit, api, both, full, dev"
    pause
    exit /b 1
)
goto :eof

:show_status
call :print_info "Checking application status..."

docker-compose ps
if errorlevel 1 docker compose ps

echo.
call :print_info "Container logs (last 10 lines):"

REM Show logs for running containers
for /f "tokens=*" %%i in ('docker ps --format "{{.Names}}" ^| findstr "banking-compliance"') do (
    echo --- Logs for %%i ---
    docker logs --tail 10 "%%i"
    echo.
)
goto :eof

:stop_application
call :print_info "Stopping application..."

docker-compose down
if errorlevel 1 docker compose down

call :print_success "Application stopped"
goto :eof

:cleanup
call :print_info "Cleaning up Docker resources..."

REM Stop and remove containers
docker-compose down -v --remove-orphans
if errorlevel 1 docker compose down -v --remove-orphans

REM Remove unused images
docker image prune -f

REM Remove unused volumes (with confirmation)
set /p answer="Do you want to remove unused volumes? This will delete persistent data. (y/N): "
if /i "%answer%"=="y" (
    docker volume prune -f
    call :print_warning "Unused volumes removed"
)

call :print_success "Cleanup completed"
goto :eof

:show_logs
set service=%~1
set lines=%~2
if "%lines%"=="" set lines=50

if "%service%"=="" (
    call :print_error "Please specify a service name"
    call :print_info "Available services: banking-compliance-app, banking-compliance-api, redis, nginx, postgres"
    pause
    exit /b 1
)

call :print_info "Showing last %lines% lines of logs for %service%..."

docker-compose logs --tail %lines% -f "%service%"
if errorlevel 1 docker compose logs --tail %lines% -f "%service%"
goto :eof

:health_check
call :print_info "Performing health check..."

set app_healthy=false
set api_healthy=false
set redis_healthy=false

REM Check Streamlit app
curl -f -s http://localhost:8501/_stcore/health >nul 2>&1
if not errorlevel 1 (
    call :print_success "Streamlit app is healthy"
    set app_healthy=true
) else (
    call :print_error "Streamlit app is not responding"
)

REM Check API
curl -f -s http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    call :print_success "API is healthy"
    set api_healthy=true
) else (
    call :print_error "API is not responding"
)

REM Check Redis
docker exec banking-compliance-redis redis-cli ping >nul 2>&1
if not errorlevel 1 (
    call :print_success "Redis is healthy"
    set redis_healthy=true
) else (
    call :print_error "Redis is not responding"
)

if "%app_healthy%"=="true" if "%api_healthy%"=="true" if "%redis_healthy%"=="true" (
    call :print_success "All services are healthy"
    exit /b 0
) else (
    call :print_error "Some services are not healthy"
    exit /b 1
)
goto :eof

:show_help
echo Banking Compliance Application Deployment Script for Windows
echo.
echo Usage: %~nx0 [COMMAND] [OPTIONS]
echo.
echo Commands:
echo   deploy [MODE]     Deploy the application
echo                     Modes: streamlit, api, both, full, dev
echo   status           Show application status
echo   logs [SERVICE]   Show logs for a service
echo   stop             Stop the application
echo   restart [MODE]   Restart the application
echo   cleanup          Stop and remove all containers and images
echo   health           Perform health check on all services
echo   setup            Setup environment configuration
echo   help             Show this help message
echo.
echo Examples:
echo   %~nx0 deploy streamlit    # Deploy only Streamlit app
echo   %~nx0 deploy both         # Deploy both Streamlit and API
echo   %~nx0 logs banking-compliance-app  # Show app logs
echo   %~nx0 status              # Show status of all services
echo   %~nx0 health              # Check health of all services
echo.
goto :eof

:main
call :print_header

if "%~1"=="" goto help
if "%~1"=="help" goto help
if "%~1"=="--help" goto help
if "%~1"=="-h" goto help

if "%~1"=="deploy" (
    call :check_prerequisites
    call :setup_environment
    call :validate_configuration
    call :build_images
    set deploy_mode=%~2
    if "!deploy_mode!"=="" set deploy_mode=streamlit
    call :deploy_application "!deploy_mode!"
    echo.
    call :show_status
    echo.
    call :health_check
) else if "%~1"=="status" (
    call :show_status
) else if "%~1"=="logs" (
    call :show_logs "%~2" "%~3"
) else if "%~1"=="stop" (
    call :stop_application
) else if "%~1"=="restart" (
    call :stop_application
    timeout /t 2 /nobreak >nul
    set restart_mode=%~2
    if "!restart_mode!"=="" set restart_mode=streamlit
    call :deploy_application "!restart_mode!"
) else if "%~1"=="cleanup" (
    call :cleanup
) else if "%~1"=="health" (
    call :health_check
) else if "%~1"=="setup" (
    call :setup_environment
    call :validate_configuration
) else (
    call :print_error "Unknown command: %~1"
    echo.
    goto help
)
goto end

:help
call :show_help
goto end

:end
pause