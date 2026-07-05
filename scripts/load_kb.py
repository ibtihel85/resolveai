"""
scripts/load_kb.py

Seeds the ChromaDB knowledge base with Meridian Insurance documents.

Run this script once before starting the application to populate
the vector store. Re-run to reset the knowledge base to its default state.

Usage:
    python scripts/load_kb.py

Prerequisites:
    - ChromaDB must be running (docker compose up chromadb)
    - CHROMA_HOST and CHROMA_PORT must be set in .env
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path so we can import src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from chromadb.utils import embedding_functions

from src.config import settings
from src.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

# ── Embedding model ───────────────────────────────────────────────────────────
# Must match the model used in knowledge_base_tool.py.
# Changing this requires re-running this script to re-embed all documents.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Knowledge base documents ──────────────────────────────────────────────────
# Each document has:
#   doc_id   — unique identifier, stable across re-seeds
#   title    — human-readable name shown in retrieval results
#   section  — category for filtering
#   text     — the content that gets embedded and searched

DOCUMENTS = [
    {
        "doc_id": "kb-001",
        "title": "Home Insurance Coverage Guide",
        "section": "coverage",
        "text": (
            "Meridian Home Insurance covers the following: fire and smoke damage, "
            "water damage from burst internal pipes or appliance leaks, theft and "
            "burglary, storm and hail damage, glass breakage, and third-party "
            "liability up to your coverage limit. "
            "Standard exclusions include: earthquake damage, groundwater flooding, "
            "gradual wear and tear, deliberate damage, and flood from external "
            "water sources such as overflowing rivers or storm surge. "
            "Optional add-ons are available for high-value items and bicycle coverage."
        ),
    },
    {
        "doc_id": "kb-002",
        "title": "How to File a Claim",
        "section": "claims_process",
        "text": (
            "To file a claim with Meridian Insurance, follow these steps: "
            "1. Report the incident within 7 days of occurrence via phone, app, or chat. "
            "2. Gather documentation — photographs of damage, receipts for lost items, "
            "and a police report if applicable for theft or vandalism. "
            "3. An adjuster will contact you within 3 business days to schedule "
            "an inspection or request additional documentation. "
            "4. You will receive a written decision within 15 business days of "
            "the completed inspection. "
            "5. Approved payments are processed within 5 business days of the decision. "
            "Filing a claim does not automatically affect your premium — this depends "
            "on your policy terms and claim history."
        ),
    },
    {
        "doc_id": "kb-003",
        "title": "Auto Insurance Coverage",
        "section": "coverage",
        "text": (
            "Meridian Auto Insurance offers four coverage types: "
            "Comprehensive coverage protects against theft, fire, weather damage, "
            "and animal collisions. "
            "Collision coverage pays for damage from accidents with other vehicles "
            "or objects regardless of fault. "
            "Third-party liability covers damage or injury you cause to others — "
            "legally required in Germany with a minimum of EUR 7.5 million for "
            "personal injury and EUR 1.12 million for property damage. "
            "Personal injury protection covers medical costs for you and your "
            "passengers regardless of fault. "
            "Your deductible applies to comprehensive and collision claims only."
        ),
    },
    {
        "doc_id": "kb-004",
        "title": "Premium Payment and Policy Renewal",
        "section": "billing",
        "text": (
            "Meridian Insurance premiums can be paid annually or monthly via "
            "SEPA direct debit. Annual payment receives a 5 percent discount. "
            "Policies renew automatically 30 days before the expiry date. "
            "A renewal notice with any premium adjustments is sent 45 days "
            "before renewal. "
            "To cancel your policy, written notice must be provided at least "
            "30 days before the renewal date. "
            "Early cancellation within the policy year results in a pro-rata "
            "refund of unused premium minus an administrative fee. "
            "Premium increases at renewal are based on claims history, "
            "regional risk data, and general market conditions."
        ),
    },
    {
        "doc_id": "kb-005",
        "title": "Understanding Your Deductible",
        "section": "policy_basics",
        "text": (
            "A deductible is the amount you pay out of pocket before your "
            "insurance coverage applies to a claim. "
            "For example, if your deductible is EUR 500 and you have a covered "
            "loss of EUR 4,500, Meridian pays EUR 4,000 and you pay EUR 500. "
            "Higher deductibles result in lower annual premiums because you "
            "absorb more of the risk. "
            "Lower deductibles result in higher premiums but less out-of-pocket "
            "cost when you file a claim. "
            "Some policy types have separate deductibles for specific claim types — "
            "for example, glass damage may carry a lower deductible than "
            "structural damage. "
            "Your deductible amount is shown on your policy declaration page."
        ),
    },
    {
        "doc_id": "kb-006",
        "title": "Flood Coverage vs Water Damage",
        "section": "coverage",
        "text": (
            "Meridian home insurance distinguishes between internal water damage "
            "and external flooding — these are covered differently. "
            "Internal water damage is covered: burst pipes, appliance leaks, "
            "roof leaks from storm damage, and accidental overflow from "
            "bathtubs or sinks. "
            "External flooding is NOT covered under standard home insurance: "
            "rising rivers, storm surge, groundwater seepage, and surface water "
            "from heavy rainfall. "
            "If your property is in a flood risk zone, a separate flood insurance "
            "add-on is available. Contact your agent to assess your flood risk "
            "and obtain a quote for supplemental coverage. "
            "Always check your policy schedule to confirm your specific coverage."
        ),
    },
    {
        "doc_id": "kb-007",
        "title": "Health Insurance Reimbursement Process",
        "section": "claims_process",
        "text": (
            "To claim reimbursement for medical expenses under your Meridian "
            "health insurance policy: "
            "Submit the original invoice and proof of payment within 3 months "
            "of the treatment date via the Meridian app, online portal, or post. "
            "In-network providers bill Meridian directly — no reimbursement "
            "submission is required for in-network care. "
            "Out-of-network reimbursement is processed within 10 business days "
            "of receiving complete documentation. "
            "Prescriptions require a valid doctor prescription to be eligible. "
            "Cosmetic procedures, experimental treatments, and self-referrals "
            "to specialists without a GP referral are not reimbursable unless "
            "specifically included in your policy schedule."
        ),
    },
    {
        "doc_id": "kb-008",
        "title": "Contacting Meridian Insurance Support",
        "section": "support",
        "text": (
            "Meridian Insurance customer support is available through multiple channels. "
            "Phone support: +49 800 123 4567, available Monday to Friday "
            "08:00 to 20:00 CET, Saturday 09:00 to 14:00 CET. "
            "AI chat support: available 24 hours a day, 7 days a week through "
            "this interface for policy questions, claims status, and general inquiries. "
            "Live agent chat: available during phone support hours for complex issues. "
            "Email: support@meridian-insurance.de with a response within "
            "1 business day. "
            "Emergency claims hotline for fire, major theft, or severe damage: "
            "+49 800 987 6543, available 24 hours a day, 7 days a week."
        ),
    },
]


def seed_knowledge_base() -> None:
    """
    Connect to ChromaDB, drop and recreate the collection,
    and index all documents with their embeddings.
    """
    log.info(
        "kb.seed_starting",
        host=settings.chroma_host,
        port=settings.chroma_port,
        collection=settings.chroma_collection_name,
        documents=len(DOCUMENTS),
    )

    # Connect to ChromaDB
    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
    )

    # Embedding function — same model used at query time in knowledge_base_tool.py
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Drop existing collection for a clean re-seed
    try:
        client.delete_collection(settings.chroma_collection_name)
        log.info("kb.collection_dropped", collection=settings.chroma_collection_name)
    except Exception:
        # Collection did not exist — nothing to drop
        pass

    # Create fresh collection with cosine similarity
    collection = client.create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Index all documents
    # ChromaDB calls the embedding function internally — we pass raw text
    collection.add(
        ids=[doc["doc_id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[
            {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "section": doc["section"],
            }
            for doc in DOCUMENTS
        ],
    )

    # Verify indexing succeeded
    count = collection.count()
    log.info(
        "kb.seed_complete",
        documents_indexed=count,
        collection=settings.chroma_collection_name,
    )
    print(f"\n✓ Knowledge base seeded: {count} documents indexed.")
    print(f"  Collection: {settings.chroma_collection_name}")
    print(f"  Host: {settings.chroma_host}:{settings.chroma_port}\n")


if __name__ == "__main__":
    seed_knowledge_base()