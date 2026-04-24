"""
FinDecide AI — FastAPI Backend
Production-grade Financial Decision Intelligence System
"""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from schemas import (
    ChatRequest, ChatResponse,
    PredictRequest, PredictResponse,
    EMIRequest, EMIResponse,
    HealthResponse, ModelStatsResponse
)
from chatbot import FinancialChatbot
from ml_engine import LoanMLEngine
from logger import get_logger

load_dotenv()
logger = get_logger(__name__)

# ── Lifespan: load ML model once at startup ───────────────────────────────────
ml_engine = LoanMLEngine()
chatbot = FinancialChatbot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FinDecide AI backend...")
    ml_engine.load_or_train()
    logger.info("ML engine ready. Starting server.")
    yield
    logger.info("Shutting down FinDecide AI backend.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinDecide AI",
    description="Financial Decision Intelligence System API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files (for single-service Render deploy)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ── Request timing middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 1)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} [{duration}ms]")
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    index = os.path.join(frontend_path, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "FinDecide AI API is running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded=ml_engine.model is not None,
        version="1.0.0"
    )


@app.get("/model/stats", response_model=ModelStatsResponse)
async def model_stats():
    stats = ml_engine.get_stats()
    if not stats:
        raise HTTPException(status_code=503, detail="Model not yet trained")
    return ModelStatsResponse(**stats)


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        result = ml_engine.predict(req.dict())
        logger.info(f"Prediction: credit={req.credit_score} income={req.income} → {result['decision']}")
        return PredictResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emi", response_model=EMIResponse)
async def calculate_emi(req: EMIRequest):
    try:
        result = ml_engine.calculate_emi(req.principal, req.annual_rate, req.months)
        return EMIResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await chatbot.respond(req.message, req.history, req.user_profile)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Graceful fallback
        return ChatResponse(
            reply="I'm having trouble connecting right now. Please try again in a moment.",
            structured_data=None,
            fallback=True
        )
