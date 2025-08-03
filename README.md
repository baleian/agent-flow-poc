### Build and Run Containers
```bash
docker-compose up --build -d
```

### Install Model
```bash
docker exec -it ollama ollama pull qwen3:8b
```

### Run langgraph GUI
```bash
cd my-app
langgraph dev
```

### Shutdown Containers
```bash
docker-compose down
```