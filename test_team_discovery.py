#!/usr/bin/env python3
"""
Test script for the improved Team Discovery V2.

Usage:
    python test_team_discovery.py planqc.eu
    python test_team_discovery.py cognizant.com "Cognizant"
    python test_team_discovery.py groeber-holzbau.de "Gröber Holzbau GmbH"
"""

import asyncio
import sys
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Clean format for visibility
)

# Reduce noise from other loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)


async def test_team_discovery(domain: str, company_name: str = None):
    """Test team discovery for a specific domain."""
    from clients.team_discovery import discover_team_contacts

    if not company_name:
        # Use domain as company name if not provided
        company_name = domain.split('.')[0].title()

    print(f"\n{'='*60}")
    print(f"TESTING TEAM DISCOVERY")
    print(f"Domain: {domain}")
    print(f"Company: {company_name}")
    print(f"{'='*60}\n")

    import time
    start = time.time()

    result = await discover_team_contacts(
        company_name=company_name,
        domain=domain
    )

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Success: {result.success}")
    print(f"Discovery Method: {result.discovery_method}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Source URLs: {result.source_urls}")
    print(f"\nContacts found: {len(result.contacts)}")

    for i, contact in enumerate(result.contacts, 1):
        print(f"\n  {i}. {contact.name}")
        if contact.title:
            print(f"     Title: {contact.title}")
        if contact.email:
            print(f"     Email: {contact.email}")
        if contact.phone:
            print(f"     Phone: {contact.phone}")

    print(f"\n{'='*60}\n")

    return result


async def run_batch_test():
    """Run tests on multiple domains."""
    test_cases = [
        ("planqc.eu", "PlanQC"),
        ("groeber-holzbau.de", "Gröber Holzbau GmbH"),
        # Add more test cases here
    ]

    results = []
    for domain, company in test_cases:
        result = await test_team_discovery(domain, company)
        results.append({
            "domain": domain,
            "company": company,
            "success": result.success,
            "contacts": len(result.contacts),
            "method": result.discovery_method
        })

    print("\n" + "="*60)
    print("BATCH RESULTS SUMMARY")
    print("="*60)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"{status} {r['domain']}: {r['contacts']} contacts ({r['method']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_team_discovery.py <domain> [company_name]")
        print("\nExamples:")
        print("  python test_team_discovery.py planqc.eu")
        print("  python test_team_discovery.py groeber-holzbau.de 'Gröber Holzbau GmbH'")
        print("  python test_team_discovery.py --batch  # Run batch tests")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        asyncio.run(run_batch_test())
    else:
        domain = sys.argv[1]
        company = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(test_team_discovery(domain, company))
