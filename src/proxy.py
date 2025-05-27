import sys

import clip
import torch
import uvicorn
from fastapi import FastAPI, Request

from src.repository import PhotoRepository

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


app = FastAPI()
repository = PhotoRepository()


@app.get("/search")
async def read_home(request: Request, search: str = "", limit: int = 10, metric: str = "COSINE"):
    text_input = clip.tokenize([search]).to(device)
    with torch.no_grad():
        features = model.encode_text(text_input)
        embedding = [float(x) for x in features.numpy()[0]]
        response = repository.search(
            embedding,
            out=["pk"],
            limit=limit,
            params={
                "metric_type": metric,
                "params": {
                    "radius": 0.2,
                    "range_filter": 1
                }
            }
        )
        return {
            "result" : str(response[0])
        }

def normalize_vector(vector):
    """
    Takes a vector and returns its normalized version

    :param vector: original vector of arbitrary length
    :return: vector of the same direction, with length 1
    """
    return vector / torch.linalg.norm(vector)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)