FROM python:3.12.12-slim

ENV PYTHONUNBUFFERED True
ENV UV_NO_CACHE True

# Set working directory
ENV APP_ROOT /root
WORKDIR $APP_ROOT

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only the requirements file first to leverage Docker cache
COPY pyproject.toml .
COPY setup.py .
COPY uv.lock .

# Install dependencies using uv
RUN uv sync --frozen --no-default-groups

# Copy only necessary files
COPY src/turtle_quant_1/ $APP_ROOT/src/turtle_quant_1/

EXPOSE 8080
CMD ["uv", "run", "uvicorn", "turtle_quant_1.server.main:app", "--host", "0.0.0.0", "--port", "8080"]