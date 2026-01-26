#!/usr/bin/env python3
"""
Test Decision Maker Discovery - Prioritized Search (HR > Dept Head > Exec)
"""
import asyncio
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test data - uhb consulting AG (ohne Ansprechpartner)
TEST_COMPANY = "uhb consulting AG"
TEST_DOMAIN = "uhb-consulting.de"
TEST_JOB_CATEGORY = "IT"  # Job category from the payload


async def test_google_dm_search_with_category():
    """Test Google Decision Maker Search with job category prioritization."""
    print("\n" + "="*60)
    print("GOOGLE DECISION MAKER SEARCH (Prioritized)")
    print("="*60)
    print(f"\nCompany: {TEST_COMPANY}")
    print(f"Domain: {TEST_DOMAIN}")
    print(f"Job Category: {TEST_JOB_CATEGORY}")
    print("\nSearch Priority:")
    print("  1. HR / Recruiting / Personalleiter")
    print("  2. IT-Leiter / CTO / Head of IT (based on category)")
    print("  3. Geschäftsführer / CEO (fallback)")

    from clients.linkedin_search import LinkedInSearchClient

    linkedin_client = LinkedInSearchClient()

    print("\n" + "-"*40)
    print("Searching...")

    result = await linkedin_client.find_decision_maker(
        company=TEST_COMPANY,
        domain=TEST_DOMAIN,
        job_category=TEST_JOB_CATEGORY
    )

    if result:
        print(f"\n✅ Found decision maker:")
        print(f"   Name:     {result['name']}")
        print(f"   Title:    {result.get('title', 'N/A')}")
        print(f"   LinkedIn: {result.get('linkedin_url', 'N/A')}")
        return result
    else:
        print("\n❌ No decision maker found")
        return None


async def main():
    print("\n" + "#"*60)
    print("# DECISION MAKER DISCOVERY TEST v2")
    print("# Priority: HR > Dept Head > Executive")
    print("#"*60)

    result = await test_google_dm_search_with_category()

    print("\n" + "="*60)
    print("RESULT")
    print("="*60)

    if result:
        print(f"\n🎯 Best Contact: {result['name']}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   LinkedIn: {result.get('linkedin_url', 'N/A')}")
    else:
        print("\n⚠️  No decision maker found - will fallback to Impressum")

    print("\n" + "#"*60)


if __name__ == "__main__":
    asyncio.run(main())
