"""
Test script for Apify LinkedIn employment verification.

Run with: python test_apify_linkedin.py

Make sure APIFY_API_KEY is set in .env file.
"""

import asyncio
import logging
from clients.apify_linkedin import get_apify_linkedin_client

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_employment_verification():
    """Test the employment verification with a real LinkedIn profile."""
    client = get_apify_linkedin_client()

    # Test cases - use real public profiles for testing
    test_cases = [
        {
            "linkedin_url": "https://www.linkedin.com/in/satlouis/",  # SAP CEO
            "expected_company": "SAP",
            "description": "SAP CEO - should be currently employed"
        },
    ]

    print("\n" + "=" * 60)
    print("APIFY LINKEDIN EMPLOYMENT VERIFICATION TEST")
    print("=" * 60)

    if not client.api_key:
        print("\n[ERROR] APIFY_API_KEY not configured in .env file!")
        print("Please add: APIFY_API_KEY=your_api_key_here")
        return

    print(f"\nAPI Key configured: {client.api_key[:10]}...")

    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"LinkedIn: {test['linkedin_url']}")
        print(f"Expected Company: {test['expected_company']}")

        try:
            result = await client.verify_employment(
                linkedin_url=test['linkedin_url'],
                expected_company=test['expected_company']
            )

            print(f"\nResult:")
            print(f"  Currently Employed: {result.is_currently_employed}")
            print(f"  Company Matched: {result.company_name_matched}")
            print(f"  Current Title: {result.current_title}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Note: {result.verification_note}")

            if result.profile:
                print(f"\n  Profile Details:")
                print(f"    Name: {result.profile.name}")
                print(f"    Headline: {result.profile.headline}")
                print(f"    Location: {result.profile.location}")
                print(f"    Experiences: {len(result.profile.experiences)}")

                for exp in result.profile.experiences[:3]:
                    current_marker = " [CURRENT]" if exp.is_current else ""
                    print(f"      - {exp.title} at {exp.company_name}{current_marker}")

        except Exception as e:
            print(f"\n[ERROR] Test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


async def test_profile_scrape_only():
    """Test just scraping a profile without verification."""
    client = get_apify_linkedin_client()

    if not client.api_key:
        print("[ERROR] APIFY_API_KEY not configured!")
        return

    # Test with a well-known public profile
    linkedin_url = "https://www.linkedin.com/in/satlouis/"

    print(f"\nScraping profile: {linkedin_url}")

    profile = await client.scrape_profile(linkedin_url)

    if profile:
        print(f"\nProfile scraped successfully!")
        print(f"  Name: {profile.name}")
        print(f"  Headline: {profile.headline}")
        print(f"  Current Company: {profile.current_company}")
        print(f"  Current Title: {profile.current_title}")
        print(f"  Total Experiences: {len(profile.experiences)}")

        print(f"\n  All Experiences:")
        for exp in profile.experiences:
            status = "[CURRENT]" if exp.is_current else f"[ENDED: {exp.end_date}]"
            print(f"    - {exp.title} at {exp.company_name} {status}")
    else:
        print("Failed to scrape profile")


if __name__ == "__main__":
    print("\nChoose test:")
    print("1. Full employment verification test")
    print("2. Profile scrape only test")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        asyncio.run(test_profile_scrape_only())
    else:
        asyncio.run(test_employment_verification())
