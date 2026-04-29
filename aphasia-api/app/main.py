from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup / shutdown lifecycle.

    Pre-loads heavy ML models at startup so the first request doesn't
    pay cold-start latency.
    """
    logger.info("Pre-loading ML models at startup …")

    # Import lazily so the module-level singletons are created now
    from app.services.wavlm_service import get_wavlm_service
    from app.services.wav2vec2_service import get_wav2vec2_service

    get_wavlm_service()
    get_wav2vec2_service()

    logger.info("All models loaded — server is ready")
    yield
    logger.info("Shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-ready API for evaluating aphasia patient speech "
            "rehabilitation using WavLM audio embeddings + DTW alignment "
            "and Wav2Vec2 CTC transcription with CER/WER text metrics."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Global exception handler ──
    @application.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on {} {}", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."},
        )

    application.include_router(router)
    return application


app = create_app()
