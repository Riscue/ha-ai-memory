import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SENTENCE_TRANSFORMER = "sentence_transformer"
FASTEMBED = "fastembed"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Embedding-Service")

app = FastAPI(title="Embedding Service (Ollama-Compatible)")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

ENGINE_TYPE = os.getenv("EMBEDDING_ENGINE", FASTEMBED).lower()
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)
CACHE_DIR = os.getenv("CACHE_DIR", "/app/cache")

loaded_models = {}


class PullRequest(BaseModel):
    name: str
    stream: bool = False  # Unused Ollama compability


class EmbedRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = None  # Unused Ollama compability
    keep_alive: Optional[str] = None  # Unused Ollama compability


def get_fastembed_model(model_name: str):
    from fastembed import TextEmbedding

    if model_name not in loaded_models:
        logger.info(f"Loading {FASTEMBED} model: {model_name}")
        try:
            loaded_models[model_name] = TextEmbedding(
                model_name=model_name,
                cache_dir=CACHE_DIR
            )
        except Exception as e:
            logger.error(f"Loading model failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    return loaded_models[model_name]


def get_sentence_transformer_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    if model_name not in loaded_models:
        logger.info(f"Loading {SENTENCE_TRANSFORMER} model: {model_name}")
        try:
            loaded_models[model_name] = SentenceTransformer(
                model_name,
                cache_folder=CACHE_DIR
            )
        except Exception as e:
            logger.error(f"Loading model failed: {e}")
            raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    return loaded_models[model_name]


def get_model(model_name: str):
    if ENGINE_TYPE == SENTENCE_TRANSFORMER:
        return get_sentence_transformer_model(model_name)
    elif ENGINE_TYPE == FASTEMBED:
        return get_fastembed_model(model_name)
    else:
        raise ValueError(f"Engine type {ENGINE_TYPE} not supported")


@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting service. Engine: {ENGINE_TYPE}, Default Model: {DEFAULT_MODEL_NAME}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        get_model(DEFAULT_MODEL_NAME)
        logger.info("Default model ready to use.")
    except Exception as e:
        logger.warning(f"Could not download default model: {e}")


@app.post("/api/pull")
async def api_pull(request: PullRequest):
    logger.info(f"Pull: {request.name}")
    get_model(request.name)
    return {"status": "success", "message": f"Model {request.name} ready"}


@app.post("/api/embed")
async def api_embeddings(request: EmbedRequest):
    logger.info(f"Embed: {request.model} - {request.prompt}")
    model = get_model(request.model)

    if ENGINE_TYPE == SENTENCE_TRANSFORMER:
        embedding = model.encode(request.prompt).tolist()
    elif ENGINE_TYPE == FASTEMBED:
        vectors = list(model.embed([request.prompt]))
        embedding = vectors[0].tolist()
    else:
        embedding = []

    return {"embedding": embedding}


@app.get("/")
def read_root():
    return {
        "status": "running",
        "engine": ENGINE_TYPE,
        "default_model": DEFAULT_MODEL_NAME
    }
