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

from backend import (
    load_data,
    train_iforest,
    detect_anomalies,
    explain_anomaly,
    analyze_sebi_query,
    CHAT_MEMORY
)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(layout="wide", page_title="Insider Trading Compliance Assistant")

# --------------------------------------------------
# Page heading
# --------------------------------------------------
st.title("Insider Trading Compliance Assistant")
st.subheader(
    "Chatbot guidance and stock-level anomaly analysis under SEBI regulations"
)

# Tabs: Chatbot FIRST, Chart SECOND
tab_chatbot, tab_chart = st.tabs(["💬 Chatbot", "📈 Market Analysis"])

# ==================================================
# CHATBOT TAB
# ==================================================
with tab_chatbot:
    st.markdown("Ask questions about SEBI insider trading rules, judgments, and compliance scenarios.")

    # 🔄 CLEAR CHAT BUTTON
    if st.button("🔄 Clear chat"):
        st.session_state.chat_history = []
        CHAT_MEMORY.clear()
        st.rerun()

    # ---- state init ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "hide_suggestions" not in st.session_state:
        st.session_state.hide_suggestions = False

    # ---- Sticky input styling ----
    st.markdown(
        """
        <style>
        div[data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            background: #0e1117;
            padding-top: 0.75rem;
            padding-bottom: 0.75rem;
            z-index: 999;
            border-top: 1px solid #2b2b2b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Render chat history ----
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # ---- Suggested prompts (before first message only) ----
    if not st.session_state.hide_suggestions:
        st.markdown("**Try asking:**")

        suggestions = [
            "What counts as UPSI under SEBI rules?",
            "Give examples of insider trading penalties",
            "Explain trading window closure",
            "Are employees allowed to trade before results?",
        ]

        cols = st.columns(len(suggestions))
        for i, text in enumerate(suggestions):
            if cols[i].button(text):
                st.session_state["prefill_query"] = text

    # ---- Chat input ----
    user_input = st.chat_input(
        "Ask about SEBI rules, judgments, insider trading…",
        key="chat_input"
    )

    if "prefill_query" in st.session_state and not user_input:
        user_input = st.session_state.pop("prefill_query")

    # ---- Handle submit ----
    if user_input:
        st.session_state.hide_suggestions = True
        st.session_state.chat_history.append(("user", user_input))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = analyze_sebi_query(user_input)

            st.markdown(result["text"])

            # ---- Case PDF downloads ----
            if result["case_files"]:
                st.markdown("**Download referenced case documents:**")
                for case in result["case_files"]:
                    with open(case["path"], "rb") as f:
                        st.download_button(
                            label=f"Download {case['label']}",
                            data=f,
                            file_name=f"{case['label']}.pdf",
                            mime="application/pdf"
                        )

        st.session_state.chat_history.append(("assistant", result["text"]))

    # ---- Footer warning ----
    st.markdown(
        "<div class='footer-warning'>"
        "⚠️ This chatbot may make mistakes. Always verify important regulatory or legal information. Please refrain from sharing any private/personal information"
        "</div>",
        unsafe_allow_html=True
    )

# ==================================================
# CHART TAB
# ==================================================
with tab_chart:
    st.subheader("Market anomaly analysis")
    st.markdown(
        "Enter a stock ticker (NSE/BSE), select a date range, and review highlighted anomalies. "
        "Click **Explain** next to a date for a detailed breakdown."
    )

    c1, c2, c3 = st.columns(3)
    ticker = c1.text_input("Ticker", "TCS.NS")
    start_date = c2.date_input("Start date", pd.to_datetime("2025-01-01"))
    end_date = c3.date_input("End date", pd.to_datetime(datetime.today()))

    df = load_data(ticker, start_date, end_date)

    model, scaler = train_iforest(df)
    mask, scores = detect_anomalies(df, model, scaler)

    df = df.copy()
    df["Anomaly"] = mask
    df["IF_score"] = scores
    anoms = df[df["Anomaly"]].copy()

    # -------- Price chart --------
    base = df.reset_index()[["Date", "Close", "Anomaly"]]

    line = alt.Chart(base).mark_line().encode(
        x="Date:T",
        y="Close:Q"
    )

    points = alt.Chart(base[base["Anomaly"]]).mark_circle(
        size=80,
        color="red"
    ).encode(
        x="Date:T",
        y="Close:Q"
    )

    st.altair_chart(
        (line + points).interactive().properties(height=520),
        use_container_width=True
    )

    # -------- Anomaly table with INLINE actions --------
    st.markdown("---")
    st.subheader("Detected anomalies")

    if anoms.empty:
        st.info("No anomalies detected in this range.")
    else:
        table = anoms.reset_index().rename(columns={"index": "Date"})
        table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")

        display_cols = [
            "Date",
            "Close",
            "Returns",
            "Volatility",
            "IF_score"
        ]

        search = st.text_input("Search anomalies (date or value)")

        if search:
            table = table[
                table.astype(str)
                .apply(lambda x: x.str.contains(search, case=False))
                .any(axis=1)
            ]

        # Render rows manually with Explain buttons
        for i, row in table.iterrows():
            with st.container():
                c1, c2, c3, c4, c5, c6 = st.columns([2, 2, 2, 2, 2, 1])

                c1.write(row["Date"])
                c2.write(f"{float(row['Close'].item()):.2f}")
                c3.write(f"{float(row['Returns'].item()):.4f}")
                c4.write(f"{float(row['Volatility'].item()):.4f}")
                c5.write(f"{float(row['IF_score'].item()):.4f}")

                if c6.button("Explain", key=f"explain_{row['Date']}"):
                    st.markdown("---")
                    st.subheader(f"Explain anomaly: {row['Date']}")

                    ma20 = df.loc[pd.to_datetime(row["Date"]), "MA20"]

                    close = float(row["Close"].item())
                    returns = float(row["Returns"].item())
                    volatility = float(row["Volatility"].item())

                    ctx = {
                        "date": row["Date"],
                        "close": close,
                        "returns": returns,
                        "volatility": volatility,
                        "ma_diff": close - float(ma20)
                    }



                    result = explain_anomaly(ctx)
                    st.markdown(result["text"])

    # ✅ Footer warning (place LAST inside tab)
    st.markdown(
        "<div class='footer-warning'>"
        "⚠️ This analysis is indicative only, and may make mistakes. It does not constitute legal or investment advice."
        "</div>",
        unsafe_allow_html=True
    )

