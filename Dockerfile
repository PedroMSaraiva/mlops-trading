FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	libpq-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY README.md /app/

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -c "import tomllib, subprocess, sys; \
    f = open('pyproject.toml', 'rb'); \
    data = tomllib.load(f); \
    f.close(); \
    deps = data.get('project', {}).get('dependencies', []); \
    reqs = [d.split(';')[0].strip() for d in deps]; \
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + reqs)"

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	libpq5 \
	&& rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY *.py /app/
COPY data/ /app/data/
COPY models/ /app/models/

RUN pip install uv

COPY pyproject.toml /app/
COPY uv.lock /app/
RUN uv sync --frozen --no-dev


RUN if [ ! -f /app/models/eth_price_predictor.pkl ] || [ ! -f /app/models/ethusdt_price_predictor.pkl ]; then \
    echo "ERRO: Modelos nao encontrados no container!"; \
    ls -lah /app/models/ || echo "Pasta models/ nao existe"; \
    exit 1; \
  else \
    echo "Modelos encontrados no container:"; \
    ls -lah /app/models/; \
  fi

RUN useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
