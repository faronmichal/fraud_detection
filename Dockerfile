FROM python:3.12-slim
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
ENV PYTHONPATH="/app"
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]