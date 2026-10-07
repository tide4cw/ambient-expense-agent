FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

RUN pip install --no-cache-dir hatchling

COPY pyproject.toml README.md main.py ./
RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
