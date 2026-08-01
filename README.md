# ResolveAI — Enterprise Insurance Support Agent

> A production-grade, multi-channel AI customer service agent for insurance — demonstrating the full lifecycle of an enterprise AI agent deployment: conversation design, prompt engineering, tool integrations, voice with SSML, guardrails, evaluation, analytics, and CI/CD.

![CI](https://github.com/ibtihel85/resolveai/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What this is

ResolveAI simulates what a Professional Services AI Agent team builds for an enterprise insurance client. The agent — **Aria** — handles customer service via chat and voice:

- Policy information and coverage questions
- Claims status lookups
- Billing and payment queries
- Callback scheduling with Google Calendar
- Escalation to human agents via Zendesk + Slack

Built with a focus on **production patterns**: prompt versioning, tool-calling, guardrails, PII redaction, evaluation harness, structured logging, and observability — not just a chatbot demo.

---

## Architecture

**Five layers, each with a clear responsibility:**

**Channel layer** — Chat API and Twilio Voice both feed into the same backend agent loop.

**Orchestration core** — Conversation manager (sliding-window memory + case state), tool calling layer (6 async tools), guardrails (PII, injection, scope), and versioned prompt registry (v1/v2).

**Integration layer** — Zendesk (real), Google Calendar (real), Slack (real), Mock Policy CRM (Guidewire-style REST API).

**Data layer** — ChromaDB for knowledge base vectors, PostgreSQL for conversation logs and analytics.

**Observability** — Streamlit analytics dashboard, Prometheus metrics, structured JSON logging.
---

## Tech stack

| Layer | Technology |
|---|---|
| Agent runtime | Python 3.11 + FastAPI (async) |
| LLM | Ollama  |
| Orchestration | Custom ReAct loop (Reasoning + Acting) |
| Vector store | ChromaDB + sentence-transformers |
| Voice STT | OpenAI Whisper (local) |
| Voice TTS | ElevenLabs with SSML / pyttsx3 fallback |
| Voice channel | Twilio Programmable Voice |
| Ticketing | Zendesk REST API (real sandbox) |
| Calendar | Google Calendar API (OAuth) |
| Escalation | Slack Incoming Webhooks |
| Mock CRM | Custom FastAPI service (Guidewire-style schema) |
| Database | PostgreSQL (conversation logs + analytics) |
| Dashboard | Streamlit + Plotly |
| Monitoring | Prometheus + Grafana |
| Deployment | Docker Compose |
| CI/CD | GitHub Actions |

---

## Quickstart

### Prerequisites

- Docker Desktop
- Python 3.11+
- [Ollama](https://ollama.com) with `llama3.2:3b` or `llama3.1:8b`
- Conda or virtualenv

### 1. Clone and configure

```bash
git clone https://github.com/ibtihel85/resolveai.git
cd resolveai
cp .env.example .env

```

### 2. Create environment and install

```bash
conda create -n resolveai python=3.11 -y
conda activate resolveai
pip install -e ".[dev]"
python -m spacy download en_core_web_lg
```

### 3. Start infrastructure

```bash
docker compose up chromadb mock-crm postgres -d
python scripts/load_kb.py      # seed knowledge base
python scripts/seed_mock_crm.py  # verify mock CRM
```

### 4. Run the API

```bash
uvicorn src.api.main:app --reload --port 8888
```

Open `http://localhost:8888/docs` — interactive API documentation.

### 5. Send your first message

```bash
curl -X POST http://localhost:8888/v1/chat/message \
  -H "Content-Type: application/json" \
  -d '{"message": "Does my home insurance cover flooding?"}'
```

### 6. View analytics dashboard

```bash
streamlit run src/analytics/dashboard.py
```

Open `http://localhost:8501`

---

## Features

### Agent capabilities
- **Policy lookups** — retrieves live policy data from mock CRM by policy ID or customer name
- **Claims status** — checks claim status, adjuster notes, estimated resolution
- **Knowledge base RAG** — semantic search over insurance FAQ and process documents
- **Callback booking** — schedules appointments via Google Calendar
- **Escalation** — creates Zendesk tickets + Slack alerts with full conversation context

### Safety and reliability
- **Guardrails layer** — blocks prompt injection, out-of-scope topics, advice requests
- **PII redaction** — Presidio anonymizes names/emails in logs (GDPR compliance)
- **Graceful degradation** — every tool returns error dicts, never crashes the conversation
- **Retry logic** — tenacity exponential backoff on all external HTTP calls
- **Fail-fast config** — pydantic-settings validates all config at startup

### Voice channel
- **Twilio integration** — inbound call handling with TwiML
- **SSML builder** — pronunciation tuning for policy IDs, currency, insurance jargon
- **Two personas** — "professional" and "warm" voice presets
- **Local Whisper STT** — free speech-to-text, no API key required

### Evaluation and observability
- **Prompt versioning** — v1/v2 with eval scores in frontmatter
- **Golden dataset** — 15 hand-crafted conversation scenarios
- **Evaluation harness** — measures task success, escalation accuracy, tool accuracy
- **Analytics dashboard** — conversation volume, escalation reasons, prompt comparison, drill-down
- **Structured logging** — JSON logs in production, coloured in development
- **PostgreSQL analytics** — every turn logged with tokens, latency, cost, tool calls

---

## Evaluation results

Running `python -m evaluation.eval_harness --prompt-version v2`:

| Metric | v1 | v2 |
|---|---|---|
| Task success rate | baseline | 86.7% |
| Escalation accuracy | baseline | 66.7% |
| Tool accuracy | baseline | 81.8% |
| Keyword match rate | baseline | 70.0% |

---

## Project structure
## Project structure

| Folder | Contents |
|---|---|
| `src/agent/` | ReAct loop, memory, guardrails, 6 tools, prompt registry |
| `src/voice/` | Whisper STT, ElevenLabs TTS, SSML builder |
| `src/api/` | FastAPI app, chat routes, Twilio voice webhooks |
| `src/analytics/` | Conversation logger, Streamlit dashboard |
| `src/db/` | SQLAlchemy models, PostgreSQL |
| `src/mocks/` | Standalone mock Policy CRM service |
| `evaluation/` | Eval harness, golden dataset (15 scenarios) |
| `tests/` | Unit (37 tests), integration (16 tests) |
| `scripts/` | KB seeding, CRM seeding |
| `monitoring/` | Prometheus + Grafana configs |
| `.github/workflows/` | GitHub Actions CI pipeline |

---

## Running tests

```bash
# Full test suite
pytest tests/unit/ tests/integration/ -v

# Unit tests only (fast, no external dependencies)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

**69 tests, all passing.**

---

## Key design decisions

**Custom ReAct loop vs LangGraph** — implements orchestration mechanics directly rather than through a framework abstraction, making every decision point explicit and debuggable.

**Mock CRM vs direct database** — the mock service has the same REST interface shape as a real Salesforce/Guidewire integration. Swapping to a real CRM is a config change (`MOCK_CRM_URL`), not a code change.

**RAG only for unstructured knowledge** — policy numbers, claim amounts, and coverage limits always come from tool calls, never from vector search. Eliminates the most dangerous hallucination risk in financial services AI.

**Guardrails outside the LLM** — escalation and safety decisions are made by deterministic Python rules, not by the LLM's self-assessment. Defense in depth.

**Prompt versioning as files** — prompts are markdown files with YAML frontmatter tracking eval scores. Changing a prompt is a file edit, not a code change. Non-engineers can read and modify them.

---

## License

MIT — see [LICENSE](LICENSE)