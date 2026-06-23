FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

# The model is loaded from Azure Blob at startup (STORAGE_BACKEND=azure_blob).
ENV STORAGE_BACKEND=azure_blob \
    MODEL_BLOB_CONTAINER=models \
    PORT=8000

EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.serve:app --host 0.0.0.0 --port ${PORT}"]
