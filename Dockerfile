FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    freetds-dev \
    freetds-bin \
    unixodbc \
    unixodbc-dev \
    tdsodbc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .streamlit directory and config
RUN mkdir -p /root/.streamlit
RUN echo '[server]\nheadless = true\nport = $PORT\naddress = "0.0.0.0"\nenableCORS = false\n' > /root/.streamlit/config.toml

# Set environment variables
ENV PORT=8501
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health

# Start command that respects Cloud Run's PORT environment variable
CMD streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true