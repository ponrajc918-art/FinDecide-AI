"""
Claude AI chatbot integration with conversation memory,
structured output parsing, what-if analysis, and fallback handling.
"""
from email.mime import message
import os
import re
import json
from dotenv import load_dotenv
from typing import Any
from anthropic import Anthropic
from logger import get_logger
from typing import Tuple, Optional
from ml_engine import LoanMLEngine

load_dotenv()

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are FinDecide AI, a senior financial intelligence assistant embedded in a bank's loan decision platform. You combine deep financial expertise with ML model outputs to deliver clear, actionable insights.

CORE CAPABILITIES:
1. Loan approval prediction with probability and reasoning
2. Risk scoring (0–100, lower = safer)
3. EMI calculations with full amortization context
4. What-if scenario analysis ("what if income increases to X?")
5. Explainable AI — always say WHY a decision was made
6. Credit and financial education

RESPONSE RULES:
- Be conversational but precise. Keep text responses to 3–5 sentences.
- ALWAYS include a JSON block when numerical analysis is requested.
- Wrap JSON in <<<JSON>>> delimiters EXACTLY as shown below.
- After JSON, provide 2–3 specific, actionable recommendations.

JSON FORMATS:

Loan prediction:
<<<JSON>>>
{
  "type": "loan_prediction",
  "approval_probability": 72,
  "risk_score": 38,
  "decision": "APPROVED",
  "loan_amount": 1500000,
  "emi": 14250,
  "tenure_months": 240,
  "interest_rate": 8.5,
  "total_interest": 1920000,
  "factors": [
    {"label": "Credit score 720 — excellent range", "impact": "positive"},
    {"label": "Debt-to-income 28% — healthy", "impact": "positive"},
    {"label": "Stable salaried employment", "impact": "positive"}
  ],
  "conditions": "Subject to property valuation and KYC verification"
}
<<<JSON>>>

EMI only:
<<<JSON>>>
{
  "type": "emi_calc",
  "principal": 1000000,
  "rate": 8.5,
  "months": 120,
  "emi": 12398,
  "total_payment": 1487760,
  "total_interest": 487760
}
<<<JSON>>>

Risk score only:
<<<JSON>>>
{
  "type": "risk_score",
  "risk_score": 45,
  "risk_level": "MEDIUM",
  "factors": [
    {"label": "Credit score 610 — below ideal threshold", "impact": "negative"},
    {"label": "Stable salaried income", "impact": "positive"}
  ]
}
<<<JSON>>>

FINANCIAL FORMULAS:
- EMI = P × r × (1+r)^n / ((1+r)^n − 1), where r = annual_rate/12/100, n = months
- Debt-to-income = (existing_monthly_debt / monthly_income) × 100
- EMI burden = (proposed_EMI / monthly_income) × 100
- Risk score components: credit score 40%, DTI 30%, default history 20%, employment 10%
- Max DTI for approval: 50%. Min credit score: 650.
- Interest rates by type: Home 8.5%, Vehicle 9%, Personal 11.5%, Business 12%, Education 8%

WHAT-IF ANALYSIS:
When user asks "what if X changes", model the new scenario, recalculate all metrics, and show the delta compared to the baseline.

TONE: Professional, empathetic, never condescending. If a loan is rejected, always suggest a clear improvement path."""


class FinancialChatbot:
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=api_key)
        self.model ="claude-3-5-sonnet-latest"

        # ✅ ADD THIS (connect ML)
        self.ml_engine = LoanMLEngine()
        self.ml_engine.load_or_train()

    async def respond(self, message, history=None, user_profile=None):
        try:
        # ✅ Step 1: Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": message}
                ]
            )

            reply_text = response.content[0].text

        # ✅ Step 2: (optional) also use ML if financial query
            msg = message.lower()

            if any(word in msg for word in ["loan", "risk", "credit"]):
                data = {
                    "age": 30,
                    "income": 45000,
                    "employment_type": "salaried",
                    "credit_score": 580,
                    "loan_amount": 500000,
                    "loan_term": 60,
                    "existing_debt": 8000,
                    "default_history": 0
                }

                result = self.ml_engine.predict(data)

                return {
                    "reply": reply_text,
                    "structured_data": {
                        "type": "loan_prediction",
                        **result
                    },
                    "fallback": False
                }

            return {
                "reply": reply_text,
                "structured_data": None,
                "fallback": False
            }

        except Exception as e:
            return {
                "reply": f"Claude API error: {str(e)}",
                "structured_data": None,
                "fallback": True
            }
    # ── Helpers ────────────────────────────────────────────────────────────────
    def _parse(self, raw: str) -> Tuple[str, Optional[dict]]:
        """Extract <<<JSON>>> block and clean reply text."""
        pattern = r"<<<JSON>>>(.*?)<<<JSON>>>"
        match = re.search(pattern, raw, re.DOTALL)
        structured = None
        if match:
            try:
                structured = json.loads(match.group(1).strip())
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed: {e}")
            raw = re.sub(pattern, "", raw, flags=re.DOTALL).strip()
        return raw, structured

    def _fallback(self, detail: str = "") -> dict:
        msg = (
            "I'm experiencing a temporary issue and couldn't process your request. "
            "Please try again shortly. "
            f"({detail})" if detail else ""
        )
        return {"reply": msg.strip(), "structured_data": None, "fallback": True}
