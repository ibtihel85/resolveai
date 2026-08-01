"""
src/metrics.py

Prometheus metrics definitions.
Imported by both api/main.py and api/routes/chat.py.
Defined here to avoid circular imports.
"""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "resolveai_requests_total",
    "Total number of chat requests",
    ["method", "endpoint", "status"],
)

TURN_LATENCY = Histogram(
    "resolveai_turn_latency_seconds",
    "Agent turn latency in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

ESCALATION_COUNT = Counter(
    "resolveai_escalations_total",
    "Total number of escalations",
    ["reason"],
)