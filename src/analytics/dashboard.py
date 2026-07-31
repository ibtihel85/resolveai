"""
src/analytics/dashboard.py

ResolveAI Analytics Dashboard — Streamlit application.

Reads conversation logs from PostgreSQL and renders:
    - KPI summary row
    - Conversation volume over time
    - Escalation reason breakdown
    - Prompt version performance comparison
    - Conversation drill-down with turn-by-turn replay

Run with:
    streamlit run src/analytics/dashboard.py

Requires PostgreSQL to be running with conversation data.
Docker: docker compose up postgres
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

from src.config import settings

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResolveAI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Database connection ───────────────────────────────────────────────────────
@st.cache_resource
def get_engine():
    """Create database engine — cached across reruns."""
    return create_engine(settings.database_url)


def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params or {})
    except Exception as exc:
        st.error(f"Database error: {exc}")
        return pd.DataFrame()


# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("ResolveAI Analytics")
st.sidebar.markdown("---")

days = st.sidebar.slider(
    "Date range (days)",
    min_value=1,
    max_value=90,
    value=30,
)

channels = st.sidebar.multiselect(
    "Channel",
    options=["chat", "voice"],
    default=["chat", "voice"],
)

since = datetime.utcnow() - timedelta(days=days)
channel_filter = "', '".join(channels) if channels else "chat"

# ── Load data ─────────────────────────────────────────────────────────────────
conversations = query(f"""
    SELECT *
    FROM conversations
    WHERE started_at >= '{since.isoformat()}'
    AND channel IN ('{channel_filter}')
    ORDER BY started_at DESC
""")

turns = query(f"""
    SELECT t.*, c.channel, c.prompt_version
    FROM conversation_turns t
    JOIN conversations c ON t.conversation_id = c.id
    WHERE t.timestamp >= '{since.isoformat()}'
    AND c.channel IN ('{channel_filter}')
    ORDER BY t.timestamp DESC
""")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 ResolveAI — Conversation Analytics")
st.caption(
    f"Meridian Insurance AI Support Agent · "
    f"Last {days} days · "
    f"Channels: {', '.join(channels) if channels else 'none'}"
)

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown("---")

total = len(conversations)
escalated = conversations["escalated"].sum() if total > 0 else 0
escalation_rate = escalated / total if total > 0 else 0

total_turns = len(turns)
fallback_turns = turns["is_fallback"].sum() if total_turns > 0 else 0
fallback_rate = fallback_turns / total_turns if total_turns > 0 else 0

avg_latency = turns["latency_ms"].mean() if total_turns > 0 else 0
total_cost = conversations["total_cost_usd"].sum() if total > 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    label="Conversations",
    value=f"{total:,}",
)
col2.metric(
    label="Escalation rate",
    value=f"{escalation_rate:.1%}",
    delta=None,
)
col3.metric(
    label="Fallback rate",
    value=f"{fallback_rate:.1%}",
)
col4.metric(
    label="Avg turn latency",
    value=f"{avg_latency:,.0f} ms",
)
col5.metric(
    label="Total cost",
    value=f"${total_cost:.4f}",
)

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Conversation volume")

    if total > 0:
        conversations["date"] = pd.to_datetime(
            conversations["started_at"]
        ).dt.date

        daily = (
            conversations
            .groupby(["date", "channel"])
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            daily,
            x="date",
            y="count",
            color="channel",
            color_discrete_map={"chat": "#4f46e5", "voice": "#0ea5e9"},
            labels={"count": "Conversations", "date": "Date"},
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No conversation data yet.")

with col_right:
    st.subheader("Escalation reasons")

    if escalated > 0:
        reasons = (
            conversations[conversations["escalated"] == True]
            ["escalation_reason"]
            .value_counts()
            .reset_index()
        )
        reasons.columns = ["reason", "count"]

        fig = px.pie(
            reasons,
            names="reason",
            values="count",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No escalations recorded.")

st.markdown("---")

# ── Prompt version comparison ─────────────────────────────────────────────────
st.subheader("Prompt version performance")

if total > 0 and "prompt_version" in conversations.columns:
    version_stats = (
        conversations
        .groupby("prompt_version")
        .agg(
            conversations=("id", "count"),
            escalations=("escalated", "sum"),
            avg_turns=("total_turns", "mean"),
            avg_cost=("total_cost_usd", "mean"),
            avg_latency_ms=("total_latency_ms", "mean"),
        )
        .reset_index()
    )
    version_stats["escalation_rate"] = (
        version_stats["escalations"] / version_stats["conversations"]
    ).map("{:.1%}".format)
    version_stats["avg_cost"] = version_stats["avg_cost"].map("${:.5f}".format)
    version_stats["avg_turns"] = version_stats["avg_turns"].map("{:.1f}".format)
    version_stats["avg_latency_ms"] = version_stats["avg_latency_ms"].map("{:,.0f} ms".format)

    st.dataframe(
        version_stats.drop(columns=["escalations"]),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No data available.")

st.markdown("---")

# ── Conversation drill-down ───────────────────────────────────────────────────
st.subheader("Conversation drill-down")

if total > 0:
    conv_options = conversations["id"].tolist()

    def format_conv(conv_id: str) -> str:
        row = conversations[conversations["id"] == conv_id].iloc[0]
        escalated_label = "🔴 escalated" if row["escalated"] else "✅ resolved"
        return f"{conv_id[:12]}... | {row['channel']} | {escalated_label}"

    selected_id = st.selectbox(
        "Select a conversation to replay",
        options=conv_options,
        format_func=format_conv,
    )

    if selected_id:
        conv_turns = turns[turns["conversation_id"] == selected_id].copy()
        conv_turns = conv_turns.sort_values("turn_index")

        if len(conv_turns) == 0:
            st.info("No turns recorded for this conversation.")
        else:
            for _, turn in conv_turns.iterrows():
                if turn["role"] == "user":
                    with st.chat_message("user"):
                        st.write(turn["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(turn["content"])

                        # Show tool calls if any
                        if turn["tool_calls"] and turn["tool_calls"] != "null":
                            tool_calls = turn["tool_calls"]
                            if isinstance(tool_calls, str):
                                try:
                                    tool_calls = json.loads(tool_calls)
                                except Exception:
                                    tool_calls = []

                            if tool_calls:
                                with st.expander(
                                    f"🔧 {len(tool_calls)} tool call(s)"
                                ):
                                    st.json(tool_calls)

                        # Metrics row
                        m1, m2, m3, m4 = st.columns(4)
                        m1.caption(f"⏱ {turn['latency_ms']:,} ms")
                        m2.caption(
                            f"🔤 {turn['input_tokens'] + turn['output_tokens']:,} tokens"
                        )
                        if turn.get("guardrail_triggered"):
                            m3.caption(f"🛡 {turn['guardrail_triggered']}")
                        if turn.get("is_escalation"):
                            m4.caption("🔴 escalated")
                        elif turn.get("is_fallback"):
                            m4.caption("⚠️ fallback")
else:
    st.info(
        "No conversations yet. "
        "Send a message via the chat API to generate data."
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "ResolveAI Analytics · "
    f"Data from PostgreSQL · "
    f"Updated on page refresh"
)