import sys
import os
import torch
import clip
from PIL import Image

from src.repository import PhotoRepository

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

if __name__ == "__main__":
    path = sys.argv[1] # path to dataset
    repository = PhotoRepository()
    # 1 step: walk directory and past embedding to database
    for root, dirs, files in os.walk(path):
        for file in files:
            image = preprocess(Image.open(path + '/' + file)).unsqueeze(0).to(device)
            ## text = clip.tokenize(["a bird", "a dog", "a cat"]).to(device)
            with torch.no_grad():
                features = model.encode_image(image)
                embedding = [float(x) for x in features.numpy()[0]]
                repository.index(
                    [
                        {
                            "pk": f"{path}/{file}",
                            "embedding": embedding
                        }
                    ]
                )
