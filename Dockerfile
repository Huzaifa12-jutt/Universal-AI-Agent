# Dockerfile for the FastAPI backend (intended for Render / any container host).
# The Streamlit frontend is deployed separately on Streamlit Community Cloud
# and does not need this Dockerfile - see README.md for details.

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY app ./app

# Render (and most PaaS hosts) inject a $PORT env var at runtime.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
