import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
from datetime import datetime

# Define paths
MEMORY_DIR = Path("ai_core/memory")
SCRAPED_DIR = Path("ai_core/scraped_data")
ONCHAIN_DIR = Path("ai_core/onchain_data")
STATUS_PATH = Path("status/health.json")
GOALS_PATH = MEMORY_DIR / "goals.json"
STRATEGY_PATH = MEMORY_DIR / "strategy_performance.json"

# Load JSON safely
def load_json(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Save JSON safely
def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# Sidebar: Goal creation form
st.sidebar.header("Create New Goal")
new_goal = st.sidebar.text_input("Enter goal text")
if st.sidebar.button("Submit Goal"):
    goals = load_json(GOALS_PATH)
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    goals.append({"goal": new_goal, "status": "pending", "timestamp": timestamp})
    save_json(GOALS_PATH, goals)
    st.sidebar.success("Goal submitted!")

# Auto-refresh every 30 seconds
st.experimental_set_query_params(refresh=str(datetime.utcnow().timestamp()))
st.button("Refresh Dashboard")

st.title("🧠 NeuroFi Dashboard")

# Section: System Health
st.subheader("🩺 System Health")
health = load_json(STATUS_PATH)
st.json(health)

# Section: Goals
st.subheader("🎯 Goals")
goals = load_json(GOALS_PATH)
if goals:
    df_goals = pd.DataFrame(goals)
    st.dataframe(df_goals)
else:
    st.info("No goals found.")

# Section: Strategy Performance
st.subheader("📊 Strategy Performance")
strategy_data = load_json(STRATEGY_PATH)
if strategy_data:
    df_strat = pd.DataFrame(strategy_data)
    fig_score = px.line(df_strat, x="timestamp", y="score", color="strategy", title="Strategy Score Over Time")
    st.plotly_chart(fig_score)

    fig_strength = px.bar(df_strat, x="strategy", y="score", title="Signal Strength by Strategy")
    st.plotly_chart(fig_strength)
else:
    st.info("No strategy performance data available.")

# Section: Scraped Data Viewer
st.subheader("🌐 Scraped Data")
scraped_files = list(SCRAPED_DIR.glob("*.json"))
if scraped_files:
    selected_scraped = st.selectbox("Select scraped file", scraped_files)
    scraped_data = load_json(selected_scraped)
    st.write(scraped_data.get("goal", ""))
    st.write("Token Mentions:", scraped_data.get("token_mentions", {}))
    df_sentiment = pd.DataFrame(scraped_data.get("sentiment", []))
    if not df_sentiment.empty:
        fig_sentiment = px.scatter(df_sentiment, x="polarity", y="subjectivity", hover_data=["text"], title="Sentiment Analysis")
        st.plotly_chart(fig_sentiment)
else:
    st.info("No scraped data available.")

# Section: On-Chain Data Viewer
st.subheader("🔗 On-Chain Data")
onchain_files = list(ONCHAIN_DIR.glob("*.json"))
if onchain_files:
    selected_onchain = st.selectbox("Select on-chain file", onchain_files)
    onchain_data = load_json(selected_onchain)
    st.json(onchain_data)
else:
    st.info("No on-chain data available.") 
