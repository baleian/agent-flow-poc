FROM python:3.9-slim

WORKDIR /

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app
COPY ./vectorstore /vectorstore

EXPOSE 50051

CMD ["python", "-m", "app.server", "-u"]