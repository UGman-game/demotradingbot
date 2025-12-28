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

MODEL_SEBI = "mistralai/mistral-7b-instruct:free"
MODEL_ANOMALY = "mistralai/mistral-7b-instruct:free"

FIXED_CONTAMINATION = 0.01
CACHE_FILE = "explanations_cache.json"

CHAT_MEMORY = []
MAX_TURNS = 6

SYSTEM_PROMPT = """
You are an insider trading compliance analyst focused on SEBI regulations.

Your role is to explain, detect, and prevent insider trading — never to facilitate it.

There are three response modes.

1) GENERAL MODE (default)
Trigger: Definitions, explanations, SEBI rules, judgments, UPSI, disclosures.
Style:
- Calm, professional, plain English
- Bullet points allowed
- Explain concepts only to help understanding, detection, or compliance
Response length:
- Short to medium (3–8 bullets or short paragraphs)

2) SUSPICIOUS OR ILLEGAL CONDUCT MODE
Trigger: User asks how insider trading happens, methods, patterns, OTC/synthetic trades, tipping, front-running, or similar.
Rules:
- You MAY describe patterns ONLY to help recognise, detect, or prevent misconduct
- NEVER give instructions, steps, loopholes, or concealment advice
Required structure:
- Brief explanation of the concept
- Why it is illegal under SEBI (PIT Regulations, 2015)
- Common red-flags or patterns (high-level, non-actionable)
- Compliance and lawful alternatives

3) ILLEGAL REQUEST MODE (STRICT REFUSAL)
Trigger: Hiding trades, exploiting UPSI, evading detection, structuring trades to profit from UPSI.
Response format (MANDATORY):
HEADLINE: Declined request
VERDICT: Refusal — cannot assist
RATIONAL:
- One clear paragraph explaining why the request is illegal
- Redirect to lawful topics (compliance, law, penalties, prevention)

Global rules:
- Never hallucinate facts, cases, or penalties
- Never provide investment or legal advice
- Never explain evasion, concealment, or trade structuring
- Treat derivatives, OTC instruments, swaps, and synthetic positions as covered by insider trading law
- Apply “substance over form” reasoning
- Use citations ONLY when authoritative documents are actually provided
- If unsure, err on the side of refusal and compliance framing
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
# ---------------- Helpers ----------------
def extract_pdf_text(path):
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def clean_label(filename: str) -> str:
    return filename.replace(".pdf", "").strip()

def case_matches_query(case_label: str, query: str) -> bool:
    q = query.lower()
    tokens = [
        t.lower()
        for t in case_label.replace("-", " ").replace(".", " ").split()
        if len(t) > 3
    ]
    return any(tok in q for tok in tokens)

def extract_primary_case_label(query: str, case_labels: list[str]):
    """
    Select exactly one dominant case for PDF usage.
    """
    q = query.lower()
    scored = []

    for label in case_labels:
        tokens = [
            t.lower()
            for t in label.replace("-", " ").replace(".", " ").split()
            if len(t) > 4
        ]
        score = sum(1 for tok in tokens if tok in q)
        if score > 0:
            scored.append((score, label))

    if not scored:
        return None

    scored.sort(reverse=True)
    return scored[0][1]

# ---------------- Document stores ----------------
REGULATION_DOCS = []
CASE_SUMMARY_DOCS = []
CASE_PDF_DOCS = []

# ---------------- Regulations ----------------
REGULATION_DIR = 'data/regulations'

REGULATION_FILES = {
    "SEBIInsiderTradingClarification2024.pdf": "SEBI Insider Trading Clarification 2024",
    "SEBIInsiderTradingRegulationJune2024.pdf": "SEBI Insider Trading Regulations June 2024",
    "Insider Trading in Dealing Room-RCM.pdf": "Insider Trading in Dealing Room (RCM)"
}

for fname, label in REGULATION_FILES.items():
    path = os.path.join(REGULATION_DIR, fname)
    if os.path.exists(path):
        REGULATION_DOCS.append({
            "label": label,
            "content": extract_pdf_text(path)
        })

# ---------------- Excel case summaries ----------------
EXCEL_DIR = 'data/judgments_excel'

if os.path.exists(EXCEL_DIR):
    for file in os.listdir(EXCEL_DIR):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(EXCEL_DIR, file))
            for _, row in df.iterrows():
                CASE_SUMMARY_DOCS.append({
                    "label": row.get("Case Name", "SEBI Case"),
                    "content": f"""
Facts:
{row.get("Brief Facts", "")}

Outcome:
{row.get("SEBI Outcome", "")}

Remarks:
{row.get("Remarks", "")}
"""
                })

# ---------------- Case PDFs ----------------
CASE_PDF_DIR = 'data/judgments_pdf'

if os.path.exists(CASE_PDF_DIR):
    for file in os.listdir(CASE_PDF_DIR):
        if file.lower().endswith(".pdf"):
            path = os.path.join(CASE_PDF_DIR, file)
            CASE_PDF_DOCS.append({
                "label": clean_label(file),
                "path": path,
                "content": extract_pdf_text(path)
            })

# ---------------- Intent detection ----------------
def needs_regulation_rag(query: str) -> bool:
    keywords = ["sebi", "regulation", "pit", "upsi", "trading window", "compliance"]
    return any(k in query.lower() for k in keywords)

def needs_case_rag(query: str) -> bool:
    keywords = ["case", "judgment", "judgement", "vs", "v.", "order",
        "held", "penalty", "court", "writ", "petition",
        "appeal", "tribunal", "sat", "high court",
        "supreme court", "sebi order"]
    return any(k in query.lower() for k in keywords)

# ---------------- Main chatbot ----------------
def analyze_sebi_query(query: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": FEW_SHOT}
    ]

    used_sources = []
    used_case_files = []

    if needs_regulation_rag(query):
        for doc in REGULATION_DOCS:
            messages.append({
                "role": "assistant",
                "content": f"AUTHORITATIVE REGULATION:\n{doc['content']}"
            })
            used_sources.append(doc["label"])

    if needs_case_rag(query):
        for doc in CASE_SUMMARY_DOCS:
            if case_matches_query(doc["label"], query):
                messages.append({
                    "role": "assistant",
                    "content": f"AUTHORITATIVE CASE SUMMARY:\n{doc['content']}"
                })
                used_sources.append(doc["label"])

        primary_case = extract_primary_case_label(
            query,
            [doc["label"] for doc in CASE_PDF_DOCS]
        )

        if primary_case:
            for doc in CASE_PDF_DOCS:
                if doc["label"] == primary_case:
                    messages.append({
                        "role": "assistant",
                        "content": f"AUTHORITATIVE CASE DOCUMENT:\n{doc['content']}"
                    })
                    used_sources.append(doc["label"])
                    used_case_files.append(doc)

    for m in CHAT_MEMORY[-MAX_TURNS:]:
        messages.append(m)

    messages.append({"role": "user", "content": query})

    reply = call_openrouter(messages, MODEL_SEBI)

    if used_sources:
        reply += "\n\n---\n**Sources:**\n"
        reply += " ".join([f"[{s}]" for s in dict.fromkeys(used_sources)])

    CHAT_MEMORY.append({"role": "user", "content": query})
    CHAT_MEMORY.append({"role": "assistant", "content": reply})

    return {
        "text": reply,
        "case_files": used_case_files
    }

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

