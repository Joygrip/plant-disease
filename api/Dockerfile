FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY src/ src/
COPY models/ models/
COPY data/splits.json data/splits.json

# CPU-only torch — no CUDA needed in container, keeps image small
RUN uv pip install --system \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system -e ".[api]"

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
