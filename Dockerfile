# Streamlit demo of the e-motor virtual sensor (runs from the compact artifacts, no dataset needed).
#   docker build -t emotor-virtual-sensor .
#   docker run -p 8501:8501 -e GEMINI_API_KEY=... emotor-virtual-sensor
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries); then the pinned runtime deps.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY .streamlit ./.streamlit
COPY src ./src
COPY artifacts ./artifacts
COPY models/virtual_sensor.joblib ./models/virtual_sensor.joblib
COPY app.py ./

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
