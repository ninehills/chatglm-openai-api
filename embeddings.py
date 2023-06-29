#!/usr/bin/env python
# coding=utf-8
from text2vec import SentenceModel

def load_embeddings_model(model_path: str, device: str):
    if device == "gpu":
        device = "cuda"
    model = SentenceModel(model_path, max_seq_length=1024, device=device)
    return model