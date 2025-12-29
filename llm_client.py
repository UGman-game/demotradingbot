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

# 🔐 Read API key from environment (Streamlit Secrets inject this)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to Streamlit Secrets before running the app."
    )

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(messages, model, max_tokens=900, temperature=0.35, top_p=0.9, timeout=30):
    """
    Call OpenRouter and return text. On HTTP / provider errors, return a helpful string
    instead of raising, and print provider response to logs for debugging.
    """
    if not OPENROUTER_API_KEY:
        return "OpenRouter API key not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as e:
        # network-level error (timeout, DNS, connection)
        print("OpenRouter request exception:", str(e))
        return f"OpenRouter request error: {e}"

    # print provider response text for debugging if non-200
    if resp.status_code != 200:
        print("OpenRouter error response (status != 200):")
        try:
            print(resp.text)
        except Exception:
            print("<could not print resp.text>")
        # Try to parse helpful provider error message
        try:
            j = resp.json()
            err = j.get("error") or j
            # show provider message or fallback
            provider_msg = err.get("message") if isinstance(err, dict) else str(err)
            return f"OpenRouter returned error {resp.status_code}: {provider_msg}"
        except Exception:
            return f"OpenRouter returned HTTP {resp.status_code}. See logs for details."

    # Now 200 -> parse body safely
    try:
        rj = resp.json()
    except Exception as e:
        print("Failed to parse OpenRouter JSON:", e)
        print("Raw response:", resp.text)
        return "OpenRouter returned non-JSON response. Check logs."

    # defensive access to fields
    try:
        text = rj["choices"][0]["message"]["content"]
    except Exception as e:
        print("Unexpected OpenRouter response structure:", e)
        print("Full response:", rj)
        return "OpenRouter returned an unexpected response format. Check logs."

    if not text or len(text.strip()) < 50:
        return (
            "The model returned an incomplete response. "
            "This anomaly shows statistical deviation but requires "
            "additional market context to assess compliance impact."
        )

    return text.strip()

