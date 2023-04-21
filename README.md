# chatglm-openai-api

Provide OpenAI style API for  ChatGLM-6B and Chinese Embeddings Model

## Todo

- [x] Add Embeddings Model
- [ ] support ChatGLM-6B Fine-tuning model
- [x] support Cloudflare Tunnel with custom domain
- [ ] add Dockerfile

## Running in local with ngrok

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Only support one GPU
CUDA_VISIBLE_DEVICES=0 python main.py --port 8080 --llm_model chatglm-6b-int4 --tunnel ngrok

# if you want to use custom ngrok domain, you need set token and subdomain in config.toml
```

## Running with Embeddings Model

```bash
CUDA_VISIBLE_DEVICES=0 python ./main.py  --llm_model chatglm-6b-int4 --embeddings_model text2vec-large-chinese
```

## Running in background

```bash
CUDA_VISIBLE_DEVICES=0 nohup python main.py --port 8080 --llm_model chatglm-6b-int4 --tunnel ngrok > nohup.out 2>&1 &
```

## Running with cloudflare tunnel

### init cloudflare tunnel

```bash
# First, you need to install cloudflare tunnel
# https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/tunnel-guide/local/

./cloudflared tunnel login
./cloudflared tunnel create chatglm-openai-api
# chatglm-openai-api.ninehills.tech is custom domain your want to use
./cloudflared tunnel route dns chatglm-openai-api chatglm-openai-api.ninehills.tech

# local debug
# ./cloudflared tunnel --url localhost:8080/. run chatglm-openai-api


CUDA_VISIBLE_DEVICES=0 python main.py --port 8080 --llm_model chatglm-6b-int4 --tunnel cloudflare
```
