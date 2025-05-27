import sys

import clip
import torch
from PIL import Image

from src.repository import PhotoRepository

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if __name__ == "__main__":
    path = sys.argv[1]
    repository = PhotoRepository()
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    ## text = clip.tokenize(["a bird", "a dog", "a cat"]).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
        embedding = [float(x) for x in features.numpy()[0]]
        response = repository.search(
            embedding,
            out=["pk"],
            limit=1000,
            params={
                "params": {
                    "radius": 0.2,
                    "range_filter": 1
                }
            }
        )

        print(response[0])

