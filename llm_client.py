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


def call_openrouter(messages, model, max_tokens=2500, temperature=0.35, top_p=0.9):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()

    text = resp.json()["choices"][0]["message"]["content"]

    if not text or len(text.strip()) < 50:
        return (
            "The bot returned an incomplete response. This is unexpected."
            "Try again with the same query"
            "or try after sometime"
        )


    return text.strip()



