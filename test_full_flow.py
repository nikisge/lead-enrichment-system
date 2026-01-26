#!/usr/bin/env python3
"""
Full Flow Integration Test - Tests the entire enrichment pipeline
"""
import asyncio
import logging
import json
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test payload - Endress+Hauser (no contact person)
TEST_PAYLOAD = {
    "category": "IT",
    "company": "Endress+Hauser Group",
    "contact_person": {"email": "", "name": "", "phone": ""},
    "date_posted": "2025-12-15",
    "description": """At Endress+Hauser, progress happens by working together. As the global leader in measurement instrumentation, our ~17,000 employees shape the future in the field of process automation. Whether developing and realizing new technology as a team, collaborating to build instrumentation, or strengthening vital relationships with countless global industries, we work to create trusted relationships that help everyone thrive.

What is the role about?
Join Endress+Hauser's NEXUS program as an Engineering Associate and embark on a transformative journey to develop technical and business skills.

Which tasks will you perform?
- Address customer inquiries and technical issues via phone, Salesforce, and email.
- Collaborate with others to provide technical support and develop solutions for customers.
- Guide customers in selecting the right products based on their needs.

What do we expect from you?
- Bachelor's degree in an engineering or related technical field
- Internship, co-op or work experience in an engineering related and/or customer focused role""",
    "id": "test-endress-hauser-123",
    "location": "Pearland, TX, USA",
    "seen": False,
    "source": "Custom",
    "title": "Engineering Associate - NEXUS",
    "url": "https://careers.endress.com/USA/job/Pearland-Engineering-Associate-NEXUS-TX-77047/1245781301/"
}


async def test_full_pipeline():
    """Test the complete enrichment pipeline."""
    from models import WebhookPayload
    from pipeline import enrich_lead

    print("\n" + "#"*70)
    print("# FULL PIPELINE INTEGRATION TEST")
    print("#"*70)
    print(f"\nCompany: {TEST_PAYLOAD['company']}")
    print(f"Job: {TEST_PAYLOAD['title']}")
    print(f"Category: {TEST_PAYLOAD['category']}")
    print(f"Location: {TEST_PAYLOAD['location']}")
    print(f"Contact in Posting: {'Yes' if TEST_PAYLOAD['contact_person']['name'] else 'NO - will search for decision maker'}")

    # Create payload
    payload = WebhookPayload(
        category=TEST_PAYLOAD["category"],
        company=TEST_PAYLOAD["company"],
        date_posted=TEST_PAYLOAD["date_posted"],
        description=TEST_PAYLOAD["description"],
        id=TEST_PAYLOAD["id"],
        location=TEST_PAYLOAD["location"],
        title=TEST_PAYLOAD["title"],
        url=TEST_PAYLOAD["url"]
    )

    print("\n" + "="*70)
    print("RUNNING PIPELINE...")
    print("="*70)

    # Run the full pipeline
    result = await enrich_lead(payload, skip_paid_apis=False)

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n✅ Success: {result.success}")
    print(f"📍 Enrichment Path: {' → '.join(result.enrichment_path)}")

    print("\n--- COMPANY ---")
    print(f"   Name: {result.company.name}")
    print(f"   Domain: {result.company.domain}")
    print(f"   Location: {result.company.location}")

    if result.decision_maker:
        print("\n--- DECISION MAKER ---")
        print(f"   Name: {result.decision_maker.name}")
        print(f"   Title: {result.decision_maker.title or 'N/A'}")
        print(f"   Email: {result.decision_maker.email or 'N/A'}")
        print(f"   LinkedIn: {result.decision_maker.linkedin_url or 'N/A'}")
    else:
        print("\n--- DECISION MAKER ---")
        print("   ❌ No decision maker found")

    if result.phone:
        print("\n--- PHONE ---")
        print(f"   📞 Number: {result.phone.number}")
        print(f"   Type: {result.phone.type.value}")
        print(f"   Source: {result.phone.source.value}")
    else:
        print("\n--- PHONE ---")
        print("   ❌ No phone found")

    if result.emails:
        print("\n--- EMAILS ---")
        for email in result.emails[:5]:
            print(f"   📧 {email}")
    else:
        print("\n--- EMAILS ---")
        print("   ❌ No emails found")

    if result.company_intel:
        print("\n--- COMPANY INTEL ---")
        print(f"   Industry: {result.company_intel.industry or 'N/A'}")
        print(f"   Employees: {result.company_intel.employee_count or 'N/A'}")
        if result.company_intel.summary:
            print(f"   Summary: {result.company_intel.summary[:200]}...")

    print("\n" + "#"*70)
    print("# TEST COMPLETE")
    print("#"*70)

    return result


async def test_free_only():
    """Test only free services (no API credits)."""
    from models import WebhookPayload
    from pipeline import enrich_lead

    print("\n" + "#"*70)
    print("# FREE SERVICES ONLY TEST (no credits)")
    print("#"*70)

    payload = WebhookPayload(
        category=TEST_PAYLOAD["category"],
        company=TEST_PAYLOAD["company"],
        date_posted=TEST_PAYLOAD["date_posted"],
        description=TEST_PAYLOAD["description"],
        id=TEST_PAYLOAD["id"],
        location=TEST_PAYLOAD["location"],
        title=TEST_PAYLOAD["title"],
        url=TEST_PAYLOAD["url"]
    )

    print("\nRunning with skip_paid_apis=True...")
    result = await enrich_lead(payload, skip_paid_apis=True)

    print(f"\n✅ Success: {result.success}")
    print(f"📍 Path: {' → '.join(result.enrichment_path)}")

    if result.decision_maker:
        print(f"👤 Decision Maker: {result.decision_maker.name}")
    if result.phone:
        print(f"📞 Phone: {result.phone.number} ({result.phone.source.value})")
    if result.emails:
        print(f"📧 Emails: {len(result.emails)} found")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--free":
        asyncio.run(test_free_only())
    else:
        asyncio.run(test_full_pipeline())
