FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install the package and expose CLI entrypoints.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data

RUN python -m pip install --upgrade pip && \
    python -m pip install .

# Runtime writable paths used by the app.
RUN mkdir -p /app/chroma_db /app/audio_out /app/image_out /app/uploads

EXPOSE 8000

CMD ["bookbot-web"]
