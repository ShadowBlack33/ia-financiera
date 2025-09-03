
FROM python:3.11-slim

WORKDIR /app
COPY ia-financiera /app/ia-financiera

RUN pip install --no-cache-dir -r ia-financiera/requirements.txt

WORKDIR /app/ia-financiera
# Default command: run tests (safe for CI)
CMD ["pytest", "-q"]
