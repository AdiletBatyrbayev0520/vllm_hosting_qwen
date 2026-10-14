# Reuses the vLLM image for its Blackwell-compatible torch + transformers + uvicorn.
FROM vllm/vllm-openai:latest
RUN pip install --no-cache-dir "laya[serve]==0.4.7"
ENTRYPOINT ["laya-serve"]
