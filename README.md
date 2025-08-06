### Build and Run Containers
```bash
docker-compose up --build -d
```

### Install Model
```bash
docker exec -it ollama ollama pull qwen3:8b
```

### langgraph GUI
```bash
https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
http://127.0.0.1:2024/docs
```

### Shutdown Containers
```bash
docker-compose down
```