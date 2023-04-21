#!/usr/bin/env python
# coding=utf-8

from transformers import AutoModel, AutoTokenizer

def init_chatglm(model_path: str, running_device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    if running_device.upper() == "GPU":
        model = model.half().cuda()
    else:
        model = model.float()
    model.eval()
    return tokenizer, model
