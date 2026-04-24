"""Pydantic schemas for all API request/response models."""
from typing import Any, Optional
from pydantic import BaseModel, Field, validator


# ── Prediction ────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    age: int = Field(..., ge=21, le=70, example=35)
    income: float = Field(..., gt=0, example=75000)
    employment_type: str = Field(..., example="salaried")
    credit_score: int = Field(..., ge=300, le=900, example=720)
    loan_amount: float = Field(..., gt=0, example=1500000)
    loan_term: int = Field(..., gt=0, le=360, description="Months", example=240)
    existing_debt: float = Field(0, ge=0, example=8000)
    default_history: int = Field(0, ge=0, le=5, example=0)

    @validator("employment_type")
    def validate_emp(cls, v):
        valid = {"salaried", "self_employed", "business", "freelance", "retired"}
        if v.lower() not in valid:
            raise ValueError(f"employment_type must be one of {valid}")
        return v.lower()


class FactorItem(BaseModel):
    label: str
    impact: str  # "positive" | "negative" | "neutral"
    weight: Optional[float] = None


class PredictResponse(BaseModel):
    decision: str
    approval_probability: float
    risk_score: float
    interest_rate: float
    emi: float
    total_interest: float
    debt_to_income: float
    emi_burden: float
    factors: list[FactorItem]
    conditions: Optional[str] = None
    model_used: str


# ── EMI ───────────────────────────────────────────────────────────────────────
class EMIRequest(BaseModel):
    principal: float = Field(..., gt=0, example=1000000)
    annual_rate: float = Field(..., gt=0, le=50, example=8.5)
    months: int = Field(..., gt=0, le=360, example=240)


class EMIResponse(BaseModel):
    emi: float
    total_payment: float
    total_interest: float
    principal: float
    annual_rate: float
    months: float
    interest_percent_of_principal: float


# ── Chat ──────────────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list)
    user_profile: Optional[dict[str, Any]] = None


class ChatResponse(BaseModel):
    reply: str
    structured_data: Optional[dict[str, Any]] = None
    fallback: bool = False


# ── Health / Stats ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelStatsResponse(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_samples: int
    feature_count: int
    trained_at: str
