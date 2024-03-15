FROM python:3.13.0a5-slim-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=app.py


CMD ["python" ,"app.py"]