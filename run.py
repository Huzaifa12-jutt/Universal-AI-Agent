"""
Convenience launcher: runs the FastAPI backend and the Streamlit frontend
together for local development.

    python run.py

For production, run each process separately (see README.md / Dockerfile):
    uvicorn app.main:app --host 0.0.0.0 --port $PORT
    streamlit run frontend/streamlit_app.py
"""

import subprocess
import sys
import threading
import time


def run_backend():
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload", "--port", "8000"]
    )


def run_frontend():
    # Give the backend a moment to boot before the frontend's health check runs
    time.sleep(2)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"]
    )


if __name__ == "__main__":
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    frontend_thread = threading.Thread(target=run_frontend, daemon=True)

    backend_thread.start()
    frontend_thread.start()

    try:
        backend_thread.join()
        frontend_thread.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
