"""
ML Pipeline: preprocessing → feature engineering → train → evaluate → save/load.
Models: Logistic Regression, Random Forest, XGBoost (best auto-selected by ROC-AUC).
"""
import os
import math
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Optional

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from data_generator import generate as generate_dataset
from logger import get_logger

logger = get_logger(__name__)

MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")
STATS_PATH = os.path.join(MODEL_DIR, "model_stats.json")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "loan_dataset.csv")

FEATURES = [
    "age", "income", "employment_type_enc", "years_employed",
    "credit_score", "loan_amount", "loan_term", "existing_debt",
    "debt_to_income", "emi_burden", "loan_to_income", "default_history",
]
EMP_MAP = {"salaried": 0, "retired": 1, "business": 2, "self_employed": 3, "freelance": 4}

BASE_RATES = {
    "home": 8.50, "vehicle": 9.00, "personal": 11.50,
    "business": 12.00, "education": 8.00
}


class LoanMLEngine:
    def __init__(self):
        self.model = None
        self.stats: dict = {}
        os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────
    def load_or_train(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                with open(STATS_PATH) as f:
                    self.stats = json.load(f)
                logger.info(f"Model loaded: {self.stats.get('model_name')} | AUC={self.stats.get('roc_auc')}")
                return
            except Exception as e:
                logger.warning(f"Could not load saved model ({e}) — retraining.")
        self.train()

    def predict(self, data: dict) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_or_train() first.")
        feats = self._engineer(data)
        X = np.array([[feats[f] for f in FEATURES]])
        prob = float(self.model.predict_proba(X)[0][1])
        decision, rate = self._decision_logic(data, prob)
        emi_data = self.calculate_emi(data["loan_amount"], rate, data["loan_term"])
        risk = self._risk_score(data)
        factors = self._explain(data, feats, prob)
        return {
            "decision":           decision,
            "approval_probability": round(prob * 100, 1),
            "risk_score":         risk,
            "interest_rate":      rate,
            "emi":                emi_data["emi"],
            "total_interest":     emi_data["total_interest"],
            "debt_to_income":     round(feats["debt_to_income"], 2),
            "emi_burden":         round(feats["emi_burden"], 2),
            "factors":            factors,
            "conditions":         self._conditions(decision, data),
            "model_used":         self.stats.get("model_name", "ensemble"),
        }

    def calculate_emi(self, principal: float, annual_rate: float, months: int) -> dict:
        r = annual_rate / 12 / 100
        if r == 0:
            emi = principal / months
        else:
            emi = principal * r * (1 + r) ** months / ((1 + r) ** months - 1)
        total = emi * months
        interest = total - principal
        return {
            "emi":                      round(emi, 2),
            "total_payment":            round(total, 2),
            "total_interest":           round(interest, 2),
            "principal":                principal,
            "annual_rate":              annual_rate,
            "months":                   months,
            "interest_percent_of_principal": round(interest / principal * 100, 1),
        }

    def get_stats(self) -> dict:
        return self.stats

    # ── Training ───────────────────────────────────────────────────────────────
    def train(self):
        logger.info("Loading / generating dataset...")
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
        else:
            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df = generate_dataset()
            df.to_csv(DATA_PATH, index=False)
        logger.info(f"Dataset: {len(df):,} rows | approval rate: {df['loan_status'].mean()*100:.1f}%")

        df = self._preprocess(df)
        X = df[FEATURES]
        y = df["loan_status"]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        candidates = self._build_candidates()
        best_model, best_name, best_auc = None, "", 0.0
        for name, pipeline in candidates.items():
            cv_scores = cross_val_score(pipeline, X_tr, y_tr, cv=StratifiedKFold(5),
                                        scoring="roc_auc", n_jobs=-1)
            auc = cv_scores.mean()
            logger.info(f"  {name}: CV AUC={auc:.4f} ± {cv_scores.std():.4f}")
            if auc > best_auc:
                best_auc, best_name, best_model = auc, name, pipeline

        logger.info(f"Best model: {best_name} (AUC={best_auc:.4f}). Fitting on full train set...")
        best_model.fit(X_tr, y_tr)

        # Evaluation
        y_pred  = best_model.predict(X_te)
        y_proba = best_model.predict_proba(X_te)[:, 1]
        self.stats = {
            "model_name":      best_name,
            "accuracy":        round(accuracy_score(y_te, y_pred), 4),
            "precision":       round(precision_score(y_te, y_pred), 4),
            "recall":          round(recall_score(y_te, y_pred), 4),
            "f1_score":        round(f1_score(y_te, y_pred), 4),
            "roc_auc":         round(roc_auc_score(y_te, y_proba), 4),
            "training_samples": len(X_tr),
            "feature_count":   len(FEATURES),
            "trained_at":      datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Test set results:\n" + classification_report(y_te, y_pred))

        self.model = best_model
        joblib.dump(best_model, MODEL_PATH)
        with open(STATS_PATH, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Model saved → {MODEL_PATH}")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["employment_type_enc"] = df["employment_type"].map(EMP_MAP).fillna(3)
        if "years_employed" not in df.columns:
            df["years_employed"] = 5
        if "loan_to_income" not in df.columns:
            df["loan_to_income"] = (df["loan_amount"] / (df["income"] * 12)).round(2)
        df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")
        return df

    def _engineer(self, data: dict) -> dict:
        inc = float(data["income"])
        loan = float(data["loan_amount"])
        term = int(data["loan_term"])
        debt = float(data.get("existing_debt", 0))
        rate = BASE_RATES.get(data.get("loan_type", "personal"), 10.0)
        r = rate / 12 / 100
        emi = loan * r * (1 + r) ** term / ((1 + r) ** term - 1) if r else loan / term
        return {
            "age":                  data["age"],
            "income":               inc,
            "employment_type_enc":  EMP_MAP.get(data.get("employment_type", "salaried"), 3),
            "years_employed":       data.get("years_employed", 5),
            "credit_score":         data["credit_score"],
            "loan_amount":          loan,
            "loan_term":            term,
            "existing_debt":        debt,
            "debt_to_income":       round(debt / inc * 100, 2),
            "emi_burden":           round(emi / inc * 100, 2),
            "loan_to_income":       round(loan / (inc * 12), 2),
            "default_history":      data.get("default_history", 0),
        }

    def _risk_score(self, data: dict) -> float:
        cs   = (900 - data["credit_score"]) / 600 * 40
        dti  = min(data.get("existing_debt", 0) / data["income"] * 100 / 100 * 25, 25)
        defs = min(data.get("default_history", 0) * 8, 24)
        emp  = {"salaried": 0, "business": 3, "self_employed": 5, "freelance": 8, "retired": 4}
        e    = emp.get(data.get("employment_type", "salaried"), 5)
        return round(min(max(cs + dti + defs + e, 0), 100), 1)

    def _decision_logic(self, data: dict, prob: float):
        cs   = data["credit_score"]
        debt = data.get("existing_debt", 0)
        inc  = data["income"]
        defs = data.get("default_history", 0)
        dti  = debt / inc * 100

        rate_key = data.get("loan_type", "personal")
        rate = BASE_RATES.get(rate_key, 10.0)

        # Risk-adjust rate
        if cs >= 750: rate -= 0.5
        elif cs < 650: rate += 1.5
        elif cs < 700: rate += 0.75
        if dti > 40: rate += 0.5
        if defs >= 2: rate += 1.0

        if cs < 600 or defs >= 3 or dti > 80:
            return "REJECTED", rate
        if cs < 650 or dti > 60 or defs == 2:
            return "CONDITIONAL", rate
        if prob >= 0.65:
            return "APPROVED", rate
        if prob >= 0.45:
            return "CONDITIONAL", rate
        return "REJECTED", rate

    def _explain(self, data: dict, feats: dict, prob: float) -> list:
        cs  = data["credit_score"]
        dti = feats["debt_to_income"]
        emib = feats["emi_burden"]
        defs = data.get("default_history", 0)
        emp  = data.get("employment_type", "salaried")
        lti  = feats["loan_to_income"]

        factors = []

        # Credit score
        if cs >= 750:
            factors.append({"label": f"Credit score {cs} — excellent range", "impact": "positive", "weight": 0.40})
        elif cs >= 700:
            factors.append({"label": f"Credit score {cs} — good range", "impact": "positive", "weight": 0.40})
        elif cs >= 650:
            factors.append({"label": f"Credit score {cs} — fair, slightly below ideal", "impact": "neutral", "weight": 0.40})
        else:
            factors.append({"label": f"Credit score {cs} — below threshold (min 650)", "impact": "negative", "weight": 0.40})

        # DTI
        if dti <= 30:
            factors.append({"label": f"Debt-to-income {dti:.1f}% — healthy", "impact": "positive", "weight": 0.30})
        elif dti <= 50:
            factors.append({"label": f"Debt-to-income {dti:.1f}% — moderate burden", "impact": "neutral", "weight": 0.30})
        else:
            factors.append({"label": f"Debt-to-income {dti:.1f}% — exceeds 50% threshold", "impact": "negative", "weight": 0.30})

        # EMI burden
        if emib <= 30:
            factors.append({"label": f"EMI burden {emib:.1f}% of income — manageable", "impact": "positive", "weight": 0.15})
        elif emib <= 50:
            factors.append({"label": f"EMI burden {emib:.1f}% of income — high but acceptable", "impact": "neutral", "weight": 0.15})
        else:
            factors.append({"label": f"EMI burden {emib:.1f}% of income — very high risk", "impact": "negative", "weight": 0.15})

        # Employment
        emp_labels = {
            "salaried":     ("Stable salaried employment", "positive"),
            "business":     ("Business owner — variable income", "neutral"),
            "self_employed":("Self-employed — income volatility risk", "neutral"),
            "freelance":    ("Freelance — irregular income pattern", "negative"),
            "retired":      ("Retired — fixed income", "neutral"),
        }
        lbl, imp = emp_labels.get(emp, ("Unknown employment type", "neutral"))
        factors.append({"label": lbl, "impact": imp, "weight": 0.10})

        # Defaults
        if defs == 0:
            factors.append({"label": "No default history — clean credit record", "impact": "positive", "weight": 0.05})
        elif defs == 1:
            factors.append({"label": "1 minor default on record", "impact": "neutral", "weight": 0.05})
        else:
            factors.append({"label": f"{defs} defaults on record — significant risk flag", "impact": "negative", "weight": 0.05})

        # LTI
        if lti > 20:
            factors.append({"label": f"Loan-to-income multiple {lti:.1f}× — very high", "impact": "negative", "weight": 0.0})
        elif lti > 10:
            factors.append({"label": f"Loan-to-income multiple {lti:.1f}× — moderate", "impact": "neutral", "weight": 0.0})

        return factors

    def _conditions(self, decision: str, data: dict) -> Optional[str]:
        if decision == "APPROVED":
            return "Subject to property/asset valuation and KYC document verification."
        if decision == "CONDITIONAL":
            tips = []
            if data["credit_score"] < 700:
                tips.append("improve credit score to 700+")
            if data.get("existing_debt", 0) / data["income"] > 0.4:
                tips.append("reduce existing EMI obligations")
            if data.get("default_history", 0) > 0:
                tips.append("clear outstanding dues and maintain clean repayment for 6 months")
            return "Conditional approval. Recommended: " + ", ".join(tips) + "." if tips else \
                   "Conditional approval pending additional income proof."
        return "Application rejected. Please address flagged risk factors and reapply after 6 months."

    def _build_candidates(self) -> dict:
        imp = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        models = {
            "LogisticRegression": Pipeline([
                ("impute", imp), ("scale", scaler),
                ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced"))
            ]),
            "RandomForest": Pipeline([
                ("impute", imp),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=12, min_samples_leaf=5,
                    class_weight="balanced", random_state=42, n_jobs=-1
                ))
            ]),
            "GradientBoosting": Pipeline([
                ("impute", imp),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200, learning_rate=0.08, max_depth=5,
                    subsample=0.8, random_state=42
                ))
            ]),
        }
        if HAS_XGB:
            models["XGBoost"] = Pipeline([
                ("impute", imp),
                ("clf", XGBClassifier(
                    n_estimators=300, learning_rate=0.07, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=2, use_label_encoder=False,
                    eval_metric="auc", random_state=42, n_jobs=-1
                ))
            ])
        return models


# Run standalone to train and save model
if __name__ == "__main__":
    engine = LoanMLEngine()
    engine.train()
    stats = engine.get_stats()
    print("\n=== Model Training Complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
