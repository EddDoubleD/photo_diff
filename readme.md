# How to start
1. install dependency
2. Run db 
3. Run main.py
4. modify cosine.py
5. analyze result

## dependency
```bash
pip install --upgrade pip
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install pymilvus

pip install fastapi
pip install uvicorn
```
## run bd

```bash
docker-compose up -d 
docker-compose down
```

```bash
docker ps
```