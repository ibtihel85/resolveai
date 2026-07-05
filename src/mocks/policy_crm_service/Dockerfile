FROM python:3.11-slim

WORKDIR /app

# Only install what this service needs — not the full resolveai dependencies
RUN pip install --no-cache-dir fastapi uvicorn

COPY main.py .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]