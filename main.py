#!/usr/bin/env python
# coding=utf-8
import argparse
import os
import sys

import toml
import uvicorn

from context import context


def main():
    parser = argparse.ArgumentParser(
        description='Start LLM and Embeddings models as a service.')

    parser.add_argument('--config', type=str, help='Path to the config file',
                        default='config.toml')
    parser.add_argument('--llm_model', type=str, help='Choosed LLM model',
                        default='chatglm-6b-int4')
    parser.add_argument('--embeddings_model', type=str,
                        help='Choosed embeddings model, can be empty',
                        default='')
    parser.add_argument('--device', type=str,
                        help='Device to run the service, gpu/cpu/mps',
                        default='gpu')
    parser.add_argument('--gpus', type=int, help='Use how many gpus, default 1',
                        default=1)
    parser.add_argument('--port', type=int, help='Port number to run the service',
                        default=8080)
    parser.add_argument('--tunnel', type=str, help='Remote tunnel for public visit, default not set',
                        default="")

    args = parser.parse_args()

    print("> Load config and arguments...")
    print(f"Config file: {args.config}")
    print(f"Language Model: {args.llm_model}")
    print(f"Embeddings Model: {args.embeddings_model}")
    print(f"Device: {args.device}")
    print(f"GPUs: {args.gpus}")
    print(f"Port: {args.port}")
    print(f"Tunneling: {args.tunnel}")

    with open(args.config) as f:
        config = toml.load(f)
        print(f"Config: \n{config}")
        context.tokens = config['auth']['tokens']

    if args.llm_model:
        print(f"> Start LLM model {args.llm_model}")
        if args.llm_model not in config['models']['llm']:
            print(f"LLM model {args.llm_model} not found in config file")
            sys.exit(1)

        llm = config['models']['llm'][args.llm_model]
        context.llm_model_type = llm['type']
        if llm['type'] == 'chatglm':
            print(f">> Use chatglm llm model {llm['path']}")
            from chatglm import init_chatglm
            context.tokenizer, context.model = init_chatglm(
                llm['path'], args.device, args.gpus)
        elif llm['type'] == 'phoenix':
            print(f">> Use phoenix llm model {llm['path']}")
            from phoenix import init_phoenix
            context.tokenizer, context.model = init_phoenix(
                llm['path'], args.device, args.gpus)
        else:
            print(f"Unsupported LLM model type {llm['type']}")
            sys.exit(1)

    if args.embeddings_model:
        print(f"> Start Embeddings model {args.embeddings_model}")
        if args.embeddings_model not in config['models']['embeddings']:
            print(
                f"Embeddings model {args.embeddings_model} not found in config file")
            sys.exit(1)

        embeddings = config['models']['embeddings'][args.embeddings_model]
        if embeddings['type'] == 'default':
            print(f">> Use default embeddings model {embeddings['path']}")
            from embeddings import load_embeddings_model
            context.embeddings_model = load_embeddings_model(
                embeddings['path'], args.device)
        else:
            print(f"Unsupported Embeddings model type {embeddings['type']}")
            sys.exit(1)

    print("> Start API server...")
    if args.tunnel:
        print(">> Enable remote tunneling...")
        if args.tunnel not in config['tunnel']:
            print(f"Tunneling {args.tunnel} not found in config file")
            sys.exit(1)
        if args.tunnel == "ngrok":
            print(">>> Start ngrok tunneling...")
            from pyngrok import ngrok, conf
            conf.get_default().region = config['tunnel']['ngrok']['region']
            if config['tunnel']['ngrok']['token']:
                ngrok.set_auth_token(config['tunnel']['ngrok']['token'])
            subdomain = config['tunnel']['ngrok']['subdomain'] or None
            http_tunnel = ngrok.connect(args.port, subdomain=subdomain)
            print(f">> Public URL: {http_tunnel.public_url}")
        if args.tunnel == "cloudflare":
            print(">>> Start cloudflare tunnel..")
            from cloudflared import run
            command = config['tunnel']['cloudflare']['cloudflared_path'] \
                or "cloudflared"
            run(command, config['tunnel']['cloudflare']['name'], args.port)

    from app import app
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == '__main__':
    main()
