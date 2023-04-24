#!/usr/bin/env python
# coding=utf-8

from transformers import AutoModel, AutoTokenizer
from utils import load_model_on_gpus

def init_chatglm(model_path: str, running_device: str, gpus: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if running_device.upper() == "GPU":
        model = load_model_on_gpus(model_path, gpus)
    else:
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = model.float()

    model.eval()
    return tokenizer, model
