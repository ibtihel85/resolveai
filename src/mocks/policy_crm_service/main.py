"""
src/mocks/policy_crm_service/main.py

Mock Policy CRM Service — simulates a Guidewire/Salesforce-style
insurance CRM for development and testing.

This is a STANDALONE FastAPI application that runs on port 8002.
It is NOT a router inside the main ResolveAI app.

Interface contract (mirrors what a real CRM would expose):
    GET  /v1/policies?policy_id=POL-xxx     → policy details
    GET  /v1/policies?customer_id=CUST-xxx  → policy by customer
    GET  /v1/claims?claim_id=CLM-xxx        → claim details
    GET  /v1/claims?policy_id=POL-xxx       → all claims for policy
    POST /v1/seed                           → reset demo data
    GET  /health                            → health check

Authentication:
    All endpoints require X-API-Key header.
    Value must match MOCK_CRM_API_KEY env var (default: mock-crm-secret-key).

Replacing with a real CRM:
    Change MOCK_CRM_URL in .env to point at the real CRM.
    Zero code changes required in the agent or tools.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query

# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.getenv("MOCK_CRM_API_KEY", "mock-crm-secret-key")


# ── Demo data ─────────────────────────────────────────────────────────────────
# Three realistic policies covering different scenarios:
#   POL-0023412 — active home insurance (happy path)
#   POL-0039871 — active auto insurance
#   POL-0051234 — lapsed health insurance (edge case)

_POLICIES: dict[str, dict[str, Any]] = {
    "POL-0023412": {
        "policy_id": "POL-0023412",
        "customer_id": "CUST-001",
        "customer_name": "Maria Hoffmann",
        "customer_email": "m.hoffmann@example.de",
        "customer_phone": "+49 89 12345678",
        "policy_type": "home",
        "status": "active",
        "start_date": "2024-01-15",
        "end_date": "2026-01-15",
        "coverage_limit": 500000.0,
        "deductible": 500.0,
        "annual_premium": 1200.0,
        "coverage_details": (
            "Full home insurance including fire, water damage from burst pipes, "
            "theft, storm and hail damage, and third-party liability. "
            "Excludes earthquake, groundwater flooding, and gradual wear."
        ),
    },
    "POL-0039871": {
        "policy_id": "POL-0039871",
        "customer_id": "CUST-002",
        "customer_name": "Klaus Braun",
        "customer_email": "k.braun@example.de",
        "customer_phone": "+49 30 98765432",
        "policy_type": "auto",
        "status": "active",
        "start_date": "2025-03-01",
        "end_date": "2026-03-01",
        "coverage_limit": 100000.0,
        "deductible": 300.0,
        "annual_premium": 850.0,
        "coverage_details": (
            "Comprehensive auto insurance including collision, theft, "
            "fire, and third-party liability up to EUR 100,000. "
            "Roadside assistance included."
        ),
    },
    "POL-0051234": {
        "policy_id": "POL-0051234",
        "customer_id": "CUST-003",
        "customer_name": "Sophie Müller",
        "customer_email": "s.mueller@example.de",
        "customer_phone": "+49 40 11223344",
        "policy_type": "health",
        "status": "lapsed",
        "start_date": "2023-06-01",
        "end_date": "2025-06-01",
        "coverage_limit": 250000.0,
        "deductible": 200.0,
        "annual_premium": 3600.0,
        "coverage_details": (
            "Private health insurance covering inpatient, outpatient, "
            "dental, and specialist visits. Policy has lapsed — "
            "renewal required to restore coverage."
        ),
    },
}

# Index by customer_id for fast lookup
_BY_CUSTOMER: dict[str, str] = {
    p["customer_id"]: pid for pid, p in _POLICIES.items()
}

# Two realistic claims
_CLAIMS: dict[str, dict[str, Any]] = {
    "CLM-0012345": {
        "claim_id": "CLM-0012345",
        "policy_id": "POL-0023412",
        "customer_id": "CUST-001",
        "claim_type": "water_damage",
        "status": "under_review",
        "date_filed": "2026-06-20",
        "estimated_resolution_date": "2026-07-15",
        "amount_claimed": 4500.0,
        "amount_approved": None,
        "adjuster_name": "Thomas Weber",
        "adjuster_notes": (
            "Site inspection completed on 2026-06-25. "
            "Awaiting contractor repair estimate. "
            "Customer should expect a decision within 10 business days."
        ),
    },
    "CLM-0009876": {
        "claim_id": "CLM-0009876",
        "policy_id": "POL-0039871",
        "customer_id": "CUST-002",
        "claim_type": "collision",
        "status": "approved",
        "date_filed": "2026-05-10",
        "estimated_resolution_date": "2026-05-25",
        "amount_claimed": 8200.0,
        "amount_approved": 7900.0,
        "adjuster_name": "Anna Fischer",
        "adjuster_notes": (
            "Damage assessment completed. Approved minus deductible of EUR 300. "
            "Payment of EUR 7,900 processed on 2026-05-24."
        ),
    },
}

# Index claims by policy_id for fast lookup
_CLAIMS_BY_POLICY: dict[str, list[str]] = {}
for cid, claim in _CLAIMS.items():
    pid = claim["policy_id"]
    _CLAIMS_BY_POLICY.setdefault(pid, []).append(cid)


# ── Auth helper ───────────────────────────────────────────────────────────────

def _verify_api_key(x_api_key: str = Header(...)) -> None:
    """
    Verify the API key header.
    Raises 401 if the key is missing or incorrect.
    """
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Set X-API-Key header correctly.",
        )


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Mock Policy CRM Service",
    version="0.1.0",
    description=(
        "Simulates a Guidewire/Salesforce-style insurance CRM. "
        "For development and testing only. "
        "Replace MOCK_CRM_URL in .env to point at a real CRM."
    ),
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Health check — no auth required."""
    return {
        "status": "ok",
        "policies": len(_POLICIES),
        "claims": len(_CLAIMS),
    }


@app.get("/v1/policies")
async def get_policy(
    policy_id: str | None = Query(default=None),
    customer_id: str | None = Query(default=None),
    customer_name: str | None = Query(default=None),
    x_api_key: str = Header(...),
) -> dict[str, Any]:
    """
    Look up a policy by policy_id, customer_id, or customer_name.
    At least one parameter is required.
    Returns 404 if no matching policy is found.
    """
    _verify_api_key(x_api_key)

    policy = None

    if policy_id:
        policy = _POLICIES.get(policy_id)

    elif customer_id:
        pid = _BY_CUSTOMER.get(customer_id)
        if pid:
            policy = _POLICIES.get(pid)

    elif customer_name:
        # Case-insensitive name search
        name_lower = customer_name.lower()
        for p in _POLICIES.values():
            if name_lower in p["customer_name"].lower():
                policy = p
                break

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: policy_id, customer_id, customer_name.",
        )

    if not policy:
        raise HTTPException(
            status_code=404,
            detail="No policy found matching the provided details.",
        )

    return policy


@app.get("/v1/claims")
async def get_claims(
    claim_id: str | None = Query(default=None),
    policy_id: str | None = Query(default=None),
    x_api_key: str = Header(...),
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Look up a claim by claim_id, or all claims for a policy_id.
    Returns 404 if no matching claim is found.
    """
    _verify_api_key(x_api_key)

    if claim_id:
        claim = _CLAIMS.get(claim_id)
        if not claim:
            raise HTTPException(
                status_code=404,
                detail=f"No claim found with ID '{claim_id}'.",
            )
        return claim

    elif policy_id:
        claim_ids = _CLAIMS_BY_POLICY.get(policy_id, [])
        if not claim_ids:
            raise HTTPException(
                status_code=404,
                detail=f"No claims found for policy '{policy_id}'.",
            )
        return [_CLAIMS[cid] for cid in claim_ids]

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: claim_id, policy_id.",
        )


@app.post("/v1/seed")
async def seed(x_api_key: str = Header(...)) -> dict:
    """
    Reset demo data to its original state.
    Useful for test isolation — ensures a clean dataset before each test run.
    Data is stored in-memory; a service restart produces the same result.
    """
    _verify_api_key(x_api_key)
    return {
        "status": "seeded",
        "policies": len(_POLICIES),
        "claims": len(_CLAIMS),
    }