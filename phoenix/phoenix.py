#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# From: https://github.com/FreedomIntelligence/LLMZoo

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .chat import do_chat, do_chat_stream


def init_phoenix(model_path: str, device: str, num_gpus: int):
    if device == "cpu":
        kwargs = {}
    elif device == "gpu":
        kwargs = {"torch_dtype": torch.float16}
        kwargs["device_map"] = "sequential"  # This is important for not the same VRAM sizes
    else:
        raise ValueError(f"Invalid device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    model.running_device = "cuda" if device == "gpu" else "cpu"
    model.do_chat = do_chat
    model.do_chat_stream = do_chat_stream
    return tokenizer, model
