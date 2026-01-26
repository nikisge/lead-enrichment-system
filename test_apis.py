"""
API Test Script - Tests each enrichment service individually.
Run with: python test_apis.py

This script tests:
1. Apollo People Search (FREE) - Find decision makers + LinkedIn URLs
2. Kaspr (1 Credit) - LinkedIn URL → Phone/Email
3. FullEnrich (10 Credits for phone) - Name + Company → Phone/Email
"""
import asyncio
import json
import sys
from config import get_settings

# Test data
TEST_COMPANY = "uhb consulting AG"
TEST_DOMAIN = "uhb-software.com"
TEST_CONTACT_NAME = "Andreas Schneider"
TEST_LINKEDIN_URL = None  # Will be filled by Apollo


async def test_apollo():
    """Test Apollo People Search - should return decision makers with LinkedIn URLs."""
    print("\n" + "=" * 60)
    print("1. APOLLO PEOPLE SEARCH (FREE)")
    print("=" * 60)

    from clients.apollo import ApolloClient
    settings = get_settings()

    if not settings.apollo_api_key:
        print("❌ Apollo API key not configured!")
        return None

    print(f"API Key: {settings.apollo_api_key[:10]}...")

    apollo = ApolloClient()

    # Test Organization Search
    print(f"\n--- Organization Search: {TEST_COMPANY} ---")
    company_info = await apollo.search_organization(TEST_COMPANY)

    if company_info:
        print(f"✅ Company found:")
        print(f"   Name: {company_info.name}")
        print(f"   Domain: {company_info.domain}")
        print(f"   Industry: {company_info.industry}")
        print(f"   Employees: {company_info.employee_count}")
        print(f"   Phone: {company_info.phone}")
        print(f"   LinkedIn: {company_info.linkedin_url}")
    else:
        print("❌ Company not found in Apollo")

    # Test People Search
    domain = company_info.domain if company_info else TEST_DOMAIN
    print(f"\n--- People Search: {domain} ---")
    print("   Searching for: HR Manager, Personalleiter, Geschäftsführer...")

    titles = [
        "HR Manager", "Personalleiter", "Geschäftsführer",
        "Head of HR", "CEO", "Managing Director"
    ]

    people = await apollo.search_people(
        domain=domain,
        titles=titles,
        location="Germany"
    )

    linkedin_url = None
    if people:
        print(f"✅ Found {len(people)} people:")
        for i, person in enumerate(people[:5]):  # Show max 5
            print(f"\n   Person {i+1}:")
            print(f"   Name: {person.name}")
            print(f"   Title: {person.title}")
            print(f"   LinkedIn: {person.linkedin_url}")
            print(f"   Email: {person.email}")

            if person.linkedin_url and not linkedin_url:
                linkedin_url = person.linkedin_url
    else:
        print("❌ No people found")

    return linkedin_url


async def test_kaspr(linkedin_url: str = None):
    """Test Kaspr API - needs LinkedIn URL."""
    print("\n" + "=" * 60)
    print("2. KASPR (1 Credit per request)")
    print("=" * 60)

    from clients.kaspr import KasprClient
    settings = get_settings()

    if not settings.kaspr_api_key:
        print("❌ Kaspr API key not configured!")
        return

    print(f"API Key: {settings.kaspr_api_key[:10]}...")

    if not linkedin_url:
        print("\n⚠️  No LinkedIn URL available from Apollo!")
        print("   Kaspr requires a LinkedIn URL to work.")
        print("   Skipping Kaspr test...")
        return

    kaspr = KasprClient()

    print(f"\n--- Enriching LinkedIn: {linkedin_url} ---")

    result = await kaspr.enrich_by_linkedin(
        linkedin_url=linkedin_url,
        name=TEST_CONTACT_NAME
    )

    if result:
        print(f"✅ Kaspr result:")
        print(f"   Success: {result.success}")
        print(f"   Phones: {len(result.phones)}")
        for phone in result.phones:
            print(f"      - {phone.number} ({phone.type.value})")
        print(f"   Emails: {len(result.emails)}")
        for email in result.emails:
            print(f"      - {email}")
    else:
        print("❌ Kaspr returned no results")


async def test_fullenrich():
    """Test FullEnrich API - works with Name + Company (no LinkedIn needed)."""
    print("\n" + "=" * 60)
    print("3. FULLENRICH (10 Credits per phone)")
    print("=" * 60)

    from clients.fullenrich import FullEnrichClient
    settings = get_settings()

    if not settings.fullenrich_api_key:
        print("❌ FullEnrich API key not configured!")
        return

    print(f"API Key: {settings.fullenrich_api_key[:10]}...")

    fullenrich = FullEnrichClient()

    print(f"\n--- Enriching: {TEST_CONTACT_NAME} at {TEST_COMPANY} ---")
    print(f"   Domain: {TEST_DOMAIN}")

    # Parse name
    names = TEST_CONTACT_NAME.split()
    first_name = names[0]
    last_name = " ".join(names[1:]) if len(names) > 1 else ""

    result = await fullenrich.enrich(
        first_name=first_name,
        last_name=last_name,
        company_name=TEST_COMPANY,
        domain=TEST_DOMAIN,
        linkedin_url=None  # Testing without LinkedIn
    )

    if result:
        print(f"✅ FullEnrich result:")
        print(f"   Success: {result.success}")
        print(f"   Phones: {len(result.phones)}")
        for phone in result.phones:
            print(f"      - {phone.number} ({phone.type.value})")
        print(f"   Emails: {len(result.emails)}")
        for email in result.emails:
            print(f"      - {email}")
    else:
        print("❌ FullEnrich returned no results (or timed out)")


async def test_impressum():
    """Test Impressum Scraping - FREE."""
    print("\n" + "=" * 60)
    print("4. IMPRESSUM SCRAPING (FREE)")
    print("=" * 60)

    from clients.impressum import ImpressumScraper

    scraper = ImpressumScraper()

    print(f"\n--- Scraping Impressum: {TEST_DOMAIN} ---")

    result = await scraper.scrape(
        company_name=TEST_COMPANY,
        domain=TEST_DOMAIN
    )

    if result:
        print(f"✅ Impressum result:")
        print(f"   Success: {result.success}")
        print(f"   Phones: {len(result.phones)}")
        for phone in result.phones:
            print(f"      - {phone.number} ({phone.type.value})")
        print(f"   Emails: {len(result.emails)}")
        for email in result.emails:
            print(f"      - {email}")
    else:
        print("❌ Impressum scraping returned no results")


async def main():
    print("=" * 60)
    print("API TEST SUITE")
    print("=" * 60)
    print(f"\nTest Company: {TEST_COMPANY}")
    print(f"Test Domain: {TEST_DOMAIN}")
    print(f"Test Contact: {TEST_CONTACT_NAME}")

    # Check which tests to run
    run_paid = "--paid" in sys.argv or "--full" in sys.argv
    run_apollo = "--apollo" in sys.argv or run_paid
    run_kaspr = "--kaspr" in sys.argv or run_paid
    run_fullenrich = "--fullenrich" in sys.argv or run_paid

    if not any([run_apollo, run_kaspr, run_fullenrich]):
        print("\n⚠️  Running FREE tests only (Impressum)")
        print("   Use --apollo, --kaspr, --fullenrich, or --paid for paid APIs")

    linkedin_url = None

    # 1. Apollo (if requested)
    if run_apollo:
        linkedin_url = await test_apollo()
    else:
        print("\n[Skipping Apollo - use --apollo to test]")

    # 2. Kaspr (if requested and we have LinkedIn)
    if run_kaspr:
        await test_kaspr(linkedin_url)
    else:
        print("\n[Skipping Kaspr - use --kaspr to test]")

    # 3. FullEnrich (if requested)
    if run_fullenrich:
        await test_fullenrich()
    else:
        print("\n[Skipping FullEnrich - use --fullenrich to test]")

    # 4. Impressum (always free)
    await test_impressum()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
