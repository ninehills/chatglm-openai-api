#!usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List

@dataclass
class Context:
    llm_model_type: str
    model: any
    tokenizer: any
    embeddings_model: any

    tokens: List[str]


context = Context(None, None, None, None, [])

