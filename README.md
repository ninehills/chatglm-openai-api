# chatglm-openai-api

Provide OpenAI style API for  ChatGLM-6B and Chinese Embeddings Model

## Todo

- [ ] Add Embeddings Model
- [ ] support ChatGLM-6B Fine-tuning model
- [ ] support Cloudflare Tunnel with custom domain
- [ ] add Dockerfile

## Running in local with tunnel

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Only support one GPU
CUDA_VISIBLE_DEVICES=0 python main.py --port 8080 --llm_model chatglm-6b-int4 --tunnel ngrok

# if you want to use custom ngrok domain, you need set token and subdomain in config.toml
```
