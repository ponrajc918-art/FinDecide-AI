"""
Large-scale synthetic financial dataset generator.
Produces 50,000 rows with realistic Indian banking distributions,
noise, edge cases, and class imbalance (~30% rejections).
Run standalone: python data_generator.py
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

SEED = 42
N_ROWS = 50_000
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "loan_dataset.csv")


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ── Demographics ──────────────────────────────────────────────────────────
    age = rng.integers(21, 71, n)

    emp_choices = ["salaried", "self_employed", "business", "freelance", "retired"]
    emp_weights = [0.55, 0.20, 0.12, 0.08, 0.05]
    employment_type = rng.choice(emp_choices, n, p=emp_weights)

    # Income by employment type (₹ per month)
    income_map = {
        "salaried":     (45_000, 80_000,  15_000, 3_00_000),
        "self_employed":(30_000, 1_20_000, 20_000, 5_00_000),
        "business":     (60_000, 2_00_000, 25_000, 10_00_000),
        "freelance":    (20_000, 80_000,  10_000, 2_00_000),
        "retired":      (15_000, 40_000,   8_000, 80_000),
    }
    income = np.zeros(n)
    for emp, (mu, sigma, lo, hi) in income_map.items():
        mask = employment_type == emp
        raw = rng.normal(mu, sigma * 0.4, mask.sum())
        income[mask] = np.clip(raw, lo, hi)
    income = np.round(income, -2)  # round to nearest 100

    years_employed = np.clip(rng.normal(6, 4, n), 0, 40).astype(int)

    # ── Credit profile ────────────────────────────────────────────────────────
    # Bimodal: mostly 650-800, tail at 550-650 (subprime)
    good = rng.normal(730, 55, n)
    bad  = rng.normal(590, 40, n)
    mix  = rng.uniform(0, 1, n)
    credit_score = np.where(mix < 0.25, bad, good)
    credit_score = np.clip(credit_score, 300, 900).astype(int)

    default_history = rng.choice([0, 1, 2, 3], n, p=[0.70, 0.18, 0.08, 0.04])

    # ── Loan details ──────────────────────────────────────────────────────────
    loan_type = rng.choice(
        ["home", "personal", "vehicle", "business", "education"],
        n, p=[0.40, 0.25, 0.15, 0.12, 0.08]
    )
    # Loan amount correlated with income
    loan_amount = np.clip(
        income * rng.uniform(5, 25, n) + rng.normal(0, 50_000, n),
        50_000, 50_00_000
    ).round(-3)

    loan_term = rng.choice(
        [12, 24, 36, 60, 84, 120, 180, 240, 300, 360],
        n, p=[0.04, 0.06, 0.10, 0.15, 0.12, 0.12, 0.12, 0.15, 0.08, 0.06]
    )

    existing_debt = np.clip(
        income * rng.uniform(0, 0.45, n) + rng.normal(0, 2000, n),
        0, income * 0.6
    ).round(-2)

    # ── Feature engineering ───────────────────────────────────────────────────
    monthly_rate = 0.085 / 12  # assume 8.5% base
    emi = loan_amount * monthly_rate * (1 + monthly_rate) ** loan_term / \
          ((1 + monthly_rate) ** loan_term - 1)

    debt_to_income = (existing_debt / income * 100).round(2)
    emi_burden     = (emi / income * 100).round(2)
    total_burden   = debt_to_income + emi_burden

    # Loan-to-income multiple
    lti = (loan_amount / (income * 12)).round(2)

    # ── Risk score (0=safe, 100=risky) ───────────────────────────────────────
    cs_component  = ((900 - credit_score) / 600 * 40).clip(0, 40)   # credit score
    dti_component = (debt_to_income / 100 * 25).clip(0, 25)          # DTI
    def_component = (default_history * 8).clip(0, 24)                # defaults
    emp_risk_map  = {"salaried": 0, "business": 3, "self_employed": 5,
                     "freelance": 8, "retired": 4}
    emp_component = np.array([emp_risk_map.get(e, 5) for e in employment_type])
    noise         = rng.normal(0, 2, n)

    risk_score = (cs_component + dti_component + def_component + emp_component + noise).clip(0, 100).round(1)

    # ── Label (loan_status) ───────────────────────────────────────────────────
    # Rule-based probability then add ML-like noise
    p_approve = np.ones(n) * 0.75
    p_approve -= (credit_score < 650).astype(float) * 0.45
    p_approve -= (credit_score < 600).astype(float) * 0.25
    p_approve -= (total_burden > 60).astype(float) * 0.35
    p_approve -= (total_burden > 80).astype(float) * 0.30
    p_approve -= (default_history >= 2).astype(float) * 0.40
    p_approve -= (lti > 20).astype(float) * 0.20
    p_approve += (credit_score > 750).astype(float) * 0.15
    p_approve += (employment_type == "salaried").astype(float) * 0.08
    p_approve += (years_employed > 5).astype(float) * 0.05
    p_approve = p_approve.clip(0.02, 0.98)
    p_approve += rng.normal(0, 0.04, n)
    p_approve = p_approve.clip(0.01, 0.99)

    loan_status = (rng.uniform(0, 1, n) < p_approve).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "user_id":          [f"USR{str(i).zfill(6)}" for i in range(n)],
        "age":              age,
        "income":           income.astype(int),
        "employment_type":  employment_type,
        "years_employed":   years_employed,
        "credit_score":     credit_score,
        "loan_amount":      loan_amount.astype(int),
        "loan_term":        loan_term,
        "loan_type":        loan_type,
        "existing_debt":    existing_debt.astype(int),
        "emi":              emi.round(0).astype(int),
        "debt_to_income":   debt_to_income,
        "emi_burden":       emi_burden,
        "loan_to_income":   lti,
        "default_history":  default_history,
        "risk_score":       risk_score,
        "loan_status":      loan_status,   # 1=approved, 0=rejected
    })

    # Edge cases: a few borderline profiles
    n_edge = 200
    edge_idx = rng.choice(df.index, n_edge, replace=False)
    df.loc[edge_idx[:100], "credit_score"] = 650  # exactly at threshold
    df.loc[edge_idx[100:], "debt_to_income"] = 49.9  # just under 50%

    print(f"Dataset generated: {len(df):,} rows | "
          f"Approval rate: {df['loan_status'].mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}")
    print(df.describe().to_string())
