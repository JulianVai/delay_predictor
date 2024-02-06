# syntax=docker/dockerfile:1.2
FROM python:3.11
# put you docker configuration here
# Copy only files from challenge directory
WORKDIR /app

COPY /challenge/ /app/
COPY requirements.txt /app/

RUN pip install -r requirements.txt

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]