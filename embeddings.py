#!/usr/bin/env python
# coding=utf-8
from text2vec import SentenceModel

def load_embeddings_model(model_path: str):
    model = SentenceModel(model_path)
    return model