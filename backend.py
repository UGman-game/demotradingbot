import requests
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import fitz
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import streamlit as st
from datetime import datetime
import altair as alt

from llm_client import call_openrouter

MODEL_SEBI = "openai/gpt-oss-20b:free"
MODEL_ANOMALY = "openai/gpt-oss-20b:free"

FIXED_CONTAMINATION = 0.01
CACHE_FILE = "explanations_cache.json"

CHAT_MEMORY = []
MAX_TURNS = 6

SYSTEM_PROMPT = """
You are an insider trading compliance analyst focused on SEBI regulations.

There are three response modes.

1) GENERAL MODE (default)
Trigger: Definitions, explanations, SEBI rules, judgments, UPSI, disclosures.
Response: 1–3 plain sentences. No structured labels.

2) SUSPICIOUS TRADE MODE
Trigger: A specific trade event or timing is described.
Response format:
HEADLINE:
VERDICT:
RATIONAL:

3) ILLEGAL REQUEST MODE
Trigger: Hiding trades, misuse of UPSI, evasion.
Response format:
HEADLINE: Declined request
VERDICT: Refusal — cannot assist
RATIONAL: I cannot help with illegal or evasive activity.

Rules:
- Never hallucinate.
- Never give investment or legal advice.
- Never explain evasion.
- Stay within SEBI insider trading scope only.
- Use citations only if documents are provided.
"""

FEW_SHOT = """
User: What is a trading window closure?
Assistant: A trading window closure is a period when insiders are prohibited from trading because the company may be handling unpublished price-sensitive information.

User: Tell me how to hide insider trades.
Assistant:
HEADLINE: Declined request
VERDICT: Refusal — cannot assist
RATIONAL: I cannot help with illegal or evasive activity.
"""

DOCUMENTS = []

# -------------------------------------------------
# Load SEBI regulation PDFs (RAG source 1)
# -------------------------------------------------
REGULATION_DIR = 'data/regulations'

def extract_pdf_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

for fname in [
    "SEBIInsiderTradingClarification2024.pdf",
    "SEBIInsiderTradingRegulationJune2024.pdf",
]:
    path = os.path.join(REGULATION_DIR, fname)
    if os.path.exists(path):
        DOCUMENTS.append(
            f"""
AUTHORITATIVE SOURCE DOCUMENT
Type: SEBI Regulation
Citation label: [{fname}]
--- BEGIN DOCUMENT ---
{extract_pdf_text(path)}
--- END DOCUMENT ---
""".strip()
        )

# -------------------------------------------------
# Load Excel-based UPSI / case references (RAG source 2)
# -------------------------------------------------
EXCEL_DIR = 'data/judgments_excel'

if os.path.exists(EXCEL_DIR):
    for file in os.listdir(EXCEL_DIR):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(EXCEL_DIR, file))

            for _, row in df.iterrows():
                block = f"""
AUTHORITATIVE SOURCE DOCUMENT
Type: SEBI Case Summary (Excel)
Case Name: {row.get("Case Name", "")}
Company: {row.get("Company", "")}
UPSI Category: {row.get("UPSI Category", "")}
Noticee Type: {row.get("Noticee Type", "")}
Trading Type: {row.get("Trading Type", "")}
SEBI Order Date: {row.get("SEBI Order Date", "")}
Outcome: {row.get("SEBI Outcome", "")}

Facts:
{row.get("Brief Facts", "")}

Remarks:
{row.get("Remarks", "")}
"""
                DOCUMENTS.append(block.strip())

def analyze_sebi_query(query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": FEW_SHOT}
    ]

    used_sources = []

    for doc in DOCUMENTS:
        messages.append({"role": "assistant", "content": doc})

        # extract citation labels safely
        if "Citation label:" in doc:
            line = doc.split("Citation label:")[1].split("\n")[0].strip()
            used_sources.append(line)

    for m in CHAT_MEMORY[-MAX_TURNS:]:
        messages.append(m)

    messages.append({"role": "user", "content": query})

    reply = call_openrouter(messages, MODEL_SEBI)

    # ---- append citation badges cleanly ----
    if used_sources:
        unique_sources = list(dict.fromkeys(used_sources))[:6]
        citation_block = "\n\n---\n**Sources:**\n"
        citation_block += "\n".join([f"• {src}" for src in unique_sources])
        reply = reply.strip() + citation_block

    CHAT_MEMORY.append({"role": "user", "content": query})
    CHAT_MEMORY.append({"role": "assistant", "content": reply})

    return reply

# ---------------- Market data ----------------

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(20).std()
    df["MA20"] = df["Close"].rolling(20).mean()
    df = df.dropna()
    return df

def train_iforest(df):
    X = df[["Close", "Volume", "Returns", "Volatility"]]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = IsolationForest(contamination=FIXED_CONTAMINATION)
    model.fit(Xs)
    return model, scaler

def detect_anomalies(df, model, scaler):
    Xs = scaler.transform(df[["Close", "Volume", "Returns", "Volatility"]])
    pred = model.predict(Xs)
    scores = model.decision_function(Xs)
    return pred == -1, scores

# ---------------- FIXED anomaly explanation ----------------

def explain_anomaly(row: dict):
    cache = json.load(open(CACHE_FILE)) if os.path.exists(CACHE_FILE) else {}
    key = f"{row['date']}|{row['close']}|{row['returns']}|{row['volatility']}"

    if key in cache:
        return cache[key]

    system = """
You are a senior market surveillance and SEBI compliance analyst.

Your job is to explain unusual trading days using ONLY the provided data.

Rules:
- Do not accuse anyone
- Do not speculate about insider intent
- Do not give investment advice
- Use calm, regulatory language

You MUST:
1. Identify why the day stands out statistically
2. Mention plausible corporate-action explanations
3. Assess whether the behavior appears routine or potentially sensitive under SEBI norms

Your output MUST contain:
- 3–5 bullet observations
- One complete paragraph explaining the price action
- One complete paragraph on SEBI compliance context
"""

    user = f"""
Date: {row['date']}
Close price: {row['close']}
Daily return: {row['returns']}
Volatility: {row['volatility']}
Deviation from 20-day average: {row['ma_diff']}
"""

    text = call_openrouter(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        MODEL_ANOMALY
    )

    result = {
        "text": text,
        "provider": "openrouter",
        "model": MODEL_ANOMALY
    }

    cache[key] = result
    json.dump(cache, open(CACHE_FILE, "w"), indent=2)
    return result