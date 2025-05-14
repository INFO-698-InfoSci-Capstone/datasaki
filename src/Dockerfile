FROM python:3.11.9

WORKDIR /app

COPY pyproject.toml poetry.lock README.md /app/
RUN pip install poetry && poetry install --no-root
RUN poetry add arcticdb==4.5.0
RUN pip install langchain langchain-groq
COPY . /app/
ENV PYTHONPATH=/app/app/proto/

CMD ["poetry", "run", "uvicorn", "app.main:app", "--reload","--host", "0.0.0.0", "--port", "8060"]
