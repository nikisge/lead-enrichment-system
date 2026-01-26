"""Test Google LinkedIn Search."""
import asyncio
from clients.linkedin_search import LinkedInSearchClient
from config import get_settings

async def main():
    settings = get_settings()
    print(f"Google API Key: {settings.google_api_key[:20] if settings.google_api_key else 'NOT SET'}...")
    print(f"Google CSE ID: {settings.google_cse_id[:20] if settings.google_cse_id else 'NOT SET'}...")

    client = LinkedInSearchClient()

    # Test search
    print("\n--- Testing LinkedIn Search ---")
    print("Query: 'Andreas Schneider' 'uhb consulting' site:linkedin.com/in")

    result = await client.find_linkedin_profile(
        name="Andreas Schneider",
        company="uhb consulting AG"
    )

    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(main())
