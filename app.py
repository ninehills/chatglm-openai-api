#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
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


class ChatBody(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


class CompletionBody(BaseModel):
    prompt: str
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


class EmbeddingsBody(BaseModel):
    # Python 3.8 does not support str | List[str]
    input: Any
    model: Optional[str]


@app.get("/")
def read_root():
    return {"Hello": "World!"}


@app.get("/v1/models")
def get_models():
    ret = {"data": [], "object": "list"}

    if context.model:
        ret['data'].append({
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
        })
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


def generate_response(content: str, chat: bool = True):
    if chat:
        return {
            "id": "chatcmpl-77PZm95TtxE0oYLRx3cxa6HtIDI7s",
            "object": "chat.completion",
            "created": 1682000966,
            "model": "gpt-3.5-turbo-0301",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop", "index": 0}
            ]
        }
    else:
        return {
            "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
            "object": "text_completion",
            "created": 1589478378,
            "model": "text-davinci-003",
            "choices": [
                {
                "text": content,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }


def generate_stream_response_start():
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk", "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
    }



def generate_stream_response(content: str, chat: bool = True):
    if chat:
        return {
            "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
            "object": "chat.completion.chunk",
            "created": 1682004627,
            "model": "gpt-3.5-turbo-0301",
            "choices": [{"delta": {"content": content}, "index": 0, "finish_reason": None}
                        ]}
    else:
        return {
            "id":"cmpl-7GfnvmcsDmmTVbPHmTBcNqlMtaEVj",
            "object":"text_completion",
            "created":1684208299,
            "choices":[
                {
                    "text": content,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
            "model": "text-davinci-003"
        }


def generate_stream_response_stop(chat: bool = True):
    if chat:
        return {"id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
            "object": "chat.completion.chunk", "created": 1682004627,
            "model": "gpt-3.5-turbo-0301",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]
            }
    else:
        return {
            "id":"cmpl-7GfnvmcsDmmTVbPHmTBcNqlMtaEVj",
            "object":"text_completion",
            "created":1684208299,
            "choices":[
                {"text":"","index":0,"logprobs":None,"finish_reason":"stop"}],
            "model":"text-davinci-003",
        }

@app.post("/v1/embeddings")
async def embeddings(body: EmbeddingsBody, request: Request, background_tasks: BackgroundTasks):
    return do_embeddings(body, request, background_tasks)


def do_embeddings(body: EmbeddingsBody, request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(torch_gc)
    if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    if not context.embeddings_model:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Embeddings model not found!")

    embeddings = context.embeddings_model.encode(body.input)
    data = []
    if isinstance(body.input, str):
        data.append({
            "object": "embedding",
            "index": 0,
            "embedding": embeddings.tolist(),
        })
    else:
        for i, embed in enumerate(embeddings):
            data.append({
                "object": "embedding",
                "index": i,
                "embedding": embed.tolist(),
            })
    content = {
        "object": "list",
        "data": data,
        "model": "text-embedding-ada-002-v2",
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0
        }
    }
    return JSONResponse(status_code=200, content=content)


@app.post("/v1/engines/{engine}/embeddings")
async def engines_embeddings(engine: str, body: EmbeddingsBody, request: Request, background_tasks: BackgroundTasks):
    return do_embeddings(body, request, background_tasks)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatBody, request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(torch_gc)
    if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    if not context.model:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")
    question = body.messages[-1]
    if question.role == 'user':
        question = question.content
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No Question Found")

    history = []
    user_question = ''
    last_role = ''
    for message in body.messages[:-1]:
        if message.role == 'system':
            history.append((message.content, "OK"))
        elif message.role == 'user':
            if last_role == 'user' and user_question != '':
                history.append((user_question, "ok"))     
            user_question = message.content        
        elif message.role == 'assistant':
            if last_role == 'user' and user_question != '':
                history.append((user_question, message.content))
                user_question = ''   
        last_role = message.role
    if user_question != '':
        history.append((user_question, "ok"))          

    print(f"question = {question}, history = {history}")

    if body.stream:
        async def eval_llm():
            first = True
            for response in context.model.do_chat_stream(
                context.model, context.tokenizer, question, history, {
                    "temperature": body.temperature,
                    "top_p": body.top_p,
                    "max_tokens": body.max_tokens,
                }):
                if first:
                    first = False
                    yield json.dumps(generate_stream_response_start(),
                                    ensure_ascii=False)
                yield json.dumps(generate_stream_response(response), ensure_ascii=False)
            yield json.dumps(generate_stream_response_stop(), ensure_ascii=False)
            yield "[DONE]"
        return EventSourceResponse(eval_llm(), ping=10000)
    else:
        response = context.model.do_chat(context.model, context.tokenizer, question, history, {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_tokens": body.max_tokens,
        })
        return JSONResponse(content=generate_response(response))


@app.post("/v1/completions")
async def completions(body: CompletionBody, request: Request, background_tasks: BackgroundTasks):
    background_tasks.add_task(torch_gc)
    if request.headers.get("Authorization").split(" ")[1] not in context.tokens:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token is wrong!")

    if not context.model:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "LLM model not found!")
    question = body.prompt

    print(f"question = {question}")

    if body.stream:
        async def eval_llm():
            for response in context.model.do_chat_stream(
                    context.model, context.tokenizer, question, [], {
                        "temperature": body.temperature,
                        "top_p": body.top_p,
                        "max_tokens": body.max_tokens,
                    }):
                yield json.dumps(generate_stream_response(response, chat=False), ensure_ascii=False)
            yield json.dumps(generate_stream_response_stop(chat=False), ensure_ascii=False)
            yield "[DONE]"
        return EventSourceResponse(eval_llm(), ping=10000)
    else:
        response = context.model.do_chat(context.model, context.tokenizer, question, [], {
            "temperature": body.temperature,
            "top_p": body.top_p,
            "max_tokens": body.max_tokens,
        })
        return JSONResponse(content=generate_response(response, chat=False))
