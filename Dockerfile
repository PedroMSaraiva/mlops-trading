FROM python:3.13-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	libpq-dev \
	curl \
	apt-transport-https \
	ca-certificates \
	gnupg \
	&& echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
	&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - \
	&& apt-get update && apt-get install -y google-cloud-cli \
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

COPY *.py /app/
COPY data/ /app/data/
COPY models/ /app/models/

RUN echo "Models in container:" && ls -lah /app/models/

RUN useradd --create-home appuser && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
