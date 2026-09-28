import os

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

MODEL_ID = os.getenv(
    "MODEL_ID", "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

pipe = pipeline("text-classification", model=MODEL_ID, device=0)
app = FastAPI(title="Sentiment classification")


class ClassifyRequest(BaseModel):
    inputs: str | list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


@app.post("/classify")
def classify(req: ClassifyRequest):
    # e.g. [{"label": "POSITIVE", "score": 0.9998}]
    return pipe(req.inputs)
