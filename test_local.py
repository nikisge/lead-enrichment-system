"""
Local test script to verify the enrichment pipeline.

Usage:
    python test_local.py           # Test mode (no paid APIs)
    python test_local.py --full    # Full mode (uses paid APIs)
"""
import asyncio
import json
import sys
from models import WebhookPayload
from pipeline import enrich_lead, enrich_lead_test_mode

# Sample payload from your n8n webhook
TEST_PAYLOAD = {
    "category": "IT",
    "company": "uhb consulting AG",
    "date_posted": "2025-12-09",
    "description": """**Zur Verstärkung unseres Teams suchen wir** **am Standort St. Wolfgang** **zum nächstmöglichen Zeitpunkt einen**

**(Junior) Consultant / Softwareberater im Kundenservice (m/w/d)**

**Diese Aufgaben erwarten Dich:**

* Du bist **verantwortlich** für Software-Einführungen sowie die fortlaufende Betreuung bei unseren Kunden.
* Du **analysierst** und **bewertest** Anforderungen, um geeignete Ansätze für den effizienten Einsatz unserer Softwarelösungen zu konzipieren.

**Bewerbung:**

kontakt@uhb-software.com

**uhb Software GmbH**
Chiemseering 1
84427 St. Wolfgang
www.uhb-software.com

**Ihr Ansprechpartner:**
Andreas Schneider""",
    "id": "https://de.indeed.com/viewjob?jk=2ff6d5200b48edf1",
    "location": "Sankt Wolfgang, BY, DE",
    "seen": False,
    "source": "Indeed",
    "title": "(Junior) Consultant / Softwareberater im Kundenservice (m/w/d)",
    "url": "https://de.indeed.com/viewjob?jk=2ff6d5200b48edf1"
}


async def main():
    full_mode = "--full" in sys.argv

    print("=" * 60)
    print("Lead Enrichment Pipeline - Local Test")
    print("=" * 60)
    print(f"\nMode: {'FULL (paid APIs)' if full_mode else 'TEST (free only)'}")

    payload = WebhookPayload(**TEST_PAYLOAD)

    print(f"\nInput:")
    print(f"  Company: {payload.company}")
    print(f"  Job: {payload.title}")
    print(f"  Location: {payload.location}")

    print("\n" + "-" * 60)
    print("Running enrichment pipeline...")
    print("-" * 60 + "\n")

    if full_mode:
        result = await enrich_lead(payload)
    else:
        result = await enrich_lead_test_mode(payload)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(f"\nSuccess: {result.success}")
    print(f"Enrichment Path: {' -> '.join(result.enrichment_path)}")

    print(f"\n--- Company ---")
    print(f"  Name: {result.company.name}")
    print(f"  Domain: {result.company.domain}")
    print(f"  Industry: {result.company.industry}")
    print(f"  Location: {result.company.location}")
    print(f"  Main Phone: {result.company.phone}")

    if result.decision_maker:
        print(f"\n--- Decision Maker ---")
        print(f"  Name: {result.decision_maker.name}")
        print(f"  Title: {result.decision_maker.title}")
        print(f"  LinkedIn: {result.decision_maker.linkedin_url}")
        print(f"  Email: {result.decision_maker.email}")

    if result.phone:
        print(f"\n--- Phone Found ---")
        print(f"  Number: {result.phone.number}")
        print(f"  Type: {result.phone.type.value}")
        print(f"  Source: {result.phone.source.value}")
    else:
        print(f"\n--- No Phone Found ---")

    if result.emails:
        print(f"\n--- Emails ---")
        for email in result.emails:
            print(f"  - {email}")

    print("\n" + "-" * 60)
    print("Full JSON Result:")
    print("-" * 60)
    print(json.dumps(result.model_dump(), indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
