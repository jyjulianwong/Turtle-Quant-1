FROM python:3.12.9-slim

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
RUN uv sync --frozen

# Copy only necessary files
COPY turtle_quant_1/ $APP_ROOT/turtle_quant_1/

CMD ["echo", "This is the Turtle Quant 1 base image."]