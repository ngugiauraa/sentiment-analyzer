FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    MODEL_SOURCE=ngugiauraa/distilbert-sentiment140 \
    TORCH_THREADS=2

RUN useradd --create-home app
WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
COPY requirements-api.txt .
RUN pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY sentiment/ sentiment/
COPY api/ api/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# The model downloads from the Hugging Face Hub on first start. To use a local
# copy instead: docker run -v ./models/final:/model -e MODEL_SOURCE=/model ...
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
