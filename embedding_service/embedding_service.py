import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SENTENCE_TRANSFORMER = "sentence_transformer"
FASTEMBED = "fastembed"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Embedding-Service")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

ENGINE_TYPE = os.getenv("EMBEDDING_ENGINE", FASTEMBED).lower()
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)
CACHE_DIR = os.getenv("CACHE_DIR", "/tmp/embedding_service/cache")

loaded_models = {}


class PullRequest(BaseModel):
    name: str


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]


def get_fastembed_model(model_name: str):
    try:
        from fastembed import TextEmbedding
    except ImportError:
        TextEmbedding = None

    if TextEmbedding is None:
        raise HTTPException(status_code=500, detail="FastEmbed not installed")

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
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        SentenceTransformer = None

    if SentenceTransformer is None:
        raise HTTPException(status_code=500, detail="SentenceTransformer not installed")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting service. Engine: {ENGINE_TYPE}, Default Model: {DEFAULT_MODEL_NAME}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        get_model(DEFAULT_MODEL_NAME)
        logger.info("Default model ready to use.")
    except Exception as e:
        logger.warning(f"Could not download default model: {e}")
    yield


app = FastAPI(title="Embedding Service (Ollama-Compatible)", lifespan=lifespan)


@app.get("/api/version")
async def api_version():
    return {"version": "0.1.0"}


@app.get("/api/tags")
async def api_tags():
    models = []
    if ENGINE_TYPE == FASTEMBED:
        try:
            from fastembed import TextEmbedding
            # list_supported_models returns a list of dicts
            supported = TextEmbedding.list_supported_models()
            for model in supported:
                models.append({
                    "name": model["model"],
                    "model": model["model"],
                    "details": {
                        "parameter_size": str(model.get("dim", "unknown")),
                        "family": "fastembed"
                    }
                })
        except Exception as e:
            logger.error(f"Failed to list fastembed models: {e}")
            # Fallback to default if list fails
            models.append({"name": DEFAULT_MODEL, "model": DEFAULT_MODEL})

    elif ENGINE_TYPE == SENTENCE_TRANSFORMER:
        # SentenceTransformer supports many, but we can list some popular ones or the default
        popular_models = [
            "BAAI/bge-small-en-v1.5",
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
            "intfloat/multilingual-e5-small"
        ]
        for m in popular_models:
            models.append({
                "name": m,
                "model": m,
                "details": {"family": "sentence_transformer"}
            })

    return {"models": models}


@app.post("/api/pull")
async def api_pull(request: PullRequest):
    logger.info(f"Pull: {request.name}")
    get_model(request.name)
    return {"status": "success"}


@app.post("/api/embed")
async def api_embeddings(request: EmbedRequest):
    model = get_model(request.model)
    inputs = request.input if isinstance(request.input, list) else [request.input]

    logger.info(f"Embed: {model} - {inputs}")

    embeddings = []
    if ENGINE_TYPE == SENTENCE_TRANSFORMER:
        embeddings = model.encode(inputs).tolist()
    elif ENGINE_TYPE == FASTEMBED:
        vectors = list(model.embed(inputs))
        embeddings = [v.tolist() for v in vectors]

    return {
        "model": request.model,
        "embeddings": embeddings,
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
    }


@app.get("/")
def read_root():
    return {
        "status": "running",
        "engine": ENGINE_TYPE,
        "default_model": DEFAULT_MODEL_NAME
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=11434)
