#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from context import context
from utils import torch_gc

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class Message(BaseModel):
    role: str
    content: str


class Body(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.95
    top_p: Optional[float] = 0.7


@app.get("/")
def read_root():
    return {"Hello": "World!"}


@app.get("/v1/models")
def get_models():
    ret = {"data": [
        {
            "created": 1677610602,
            "id": "gpt-3.5-turbo",
            "object": "model",
            "owned_by": "openai",
            "permission": [
                {
                    "created": 1680818747,
                    "id": "modelperm-fTUZTbzFp7uLLTeMSo9ks6oT",
                    "object": "model_permission",
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ],
            "root": "gpt-3.5-turbo",
            "parent": None,
        },
    ],
        "object": "list"
    }

    if context.embeddings_model:
        ret['data'].append({
            "created": 1671217299,
            "id": "text-embedding-ada-002",
            "object": "model",
            "owned_by": "openai-internal",
            "permission": [
                {
                    "created": 1678892857,
                    "id": "modelperm-Dbv2FOgMdlDjO8py8vEjD5Mi",
                    "object": "model_permission",
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": True,
                    "allow_search_indices": True,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False
                }
            ],
            "root": "text-embedding-ada-002",
            "parent": ""
        })

    return ret


def generate_response(content: str):
    return {
        "id": "chatcmpl-77PZm95TtxE0oYLRx3cxa6HtIDI7s",
        "object": "chat.completion",
        "created": 1682000966,
        "model": "gpt-3.5-turbo-0301",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        },
        "choices": [{
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop", "index": 0}
        ]
    }


def generate_stream_response_start():
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk", "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
    }


def generate_stream_response(content: str):
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk",
        "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}
                    ]}


def generate_stream_response_stop():
    return {"id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
            "object": "chat.completion.chunk", "created": 1682004627,
            "model": "gpt-3.5-turbo-0301",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
            }


@app.post("/v1/chat/completions")
async def completions(body: Body, request: Request):
    if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    torch_gc()

    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    history = []
    user_question = ''
    for message in body.messages:
        if message.role == 'system':
            history.append((message.content, "OK"))
        if message.role == 'user':
            user_question = message.content
        elif message.role == 'assistant':
            assistant_answer = message.content
            history.append((user_question, assistant_answer))

    print(f"question = {question}, history = {history}")

    if body.stream:
        async def eval_chatglm():
            sends = 0
            first = True
            for response, _ in context.model.stream_chat(
                    context.tokenizer, question, history,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    max_length=max(2048, body.max_tokens)):
                if await request.is_disconnected():
                    return
                ret = response[sends:]
                sends = len(response)
                if first:
                    first = False
                    yield json.dumps(generate_stream_response_start(),
                                     ensure_ascii=False)
                yield json.dumps(generate_stream_response(ret), ensure_ascii=False)
            yield json.dumps(generate_stream_response_stop(), ensure_ascii=False)
            yield "[DONE]"
        return EventSourceResponse(eval_chatglm(), ping=10000)
    else:
        response, _ = context.model.chat(
            context.tokenizer, question, history,
            temperature=body.temperature,
            top_p=body.top_p,
            max_length=max(2048, body.max_tokens))
        print(f"response: {response}")
        return JSONResponse(content=generate_response(response))

