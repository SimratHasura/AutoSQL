FROM tiangolo/uvicorn-gunicorn-fastapi:python3.11

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

COPY ./api/demo /app
COPY pyproject.toml poetry.lock /app/

ENV PYTHONPATH=/app
RUN pip install "poetry==1.4.2"
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-interaction --no-ansi
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]