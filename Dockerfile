FROM python:3.13-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	libpq-dev \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/
COPY README.md /app/

RUN python -m pip install --upgrade pip setuptools wheel

# Instalar dependências do pyproject.toml
RUN python -c "import tomllib, subprocess, sys; \
    f = open('pyproject.toml', 'rb'); \
    data = tomllib.load(f); \
    f.close(); \
    deps = data.get('project', {}).get('dependencies', []); \
    reqs = [d.split(';')[0].strip() for d in deps]; \
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + reqs)"

COPY . /app

RUN useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
