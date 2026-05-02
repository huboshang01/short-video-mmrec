import sys
import torch
import pandas as pd
import numpy as np
import sklearn
import faiss
from sentence_transformers import SentenceTransformer

print("=" * 60)
print("V1 Environment Check")
print("=" * 60)
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Pandas:", pd.__version__)
print("Numpy:", np.__version__)
print("Sklearn:", sklearn.__version__)
print("Faiss: OK")
print("SentenceTransformers: OK")
print("=" * 60)
print("V1 env ready.")