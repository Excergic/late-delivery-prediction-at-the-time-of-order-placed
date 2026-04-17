FROM python:3.11-slim

WORKDIR /app

# libgomp1 is required by LightGBM (OpenMP runtime) — missing from slim base image
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (layer cached unless requirements change)
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs at runtime
COPY api/          api/
COPY core/         core/
COPY model/        model/
COPY configs/      configs/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
