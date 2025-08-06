FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./my_agent /app/my_agent
COPY ./.env /app/.env
COPY ./langgraph.json /app/langgraph.json

EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0"]
