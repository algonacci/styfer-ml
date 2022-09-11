FROM python:3.10-slim

ENV PYTHONBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . ./

RUN apt-get update && apt-get install -y protobuf-compiler

RUN pip install -r requirements.txt

RUN pip install protobuf

RUN PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app