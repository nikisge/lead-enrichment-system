import logging
import httpx
from typing import Optional, List

from config import get_settings
from models import DecisionMaker, CompanyInfo

logger = logging.getLogger(__name__)

APOLLO_BASE_URL = "https://api.apollo.io/api/v1"


class ApolloClient:
    """Apollo.io API client for people discovery and company search."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.apollo_api_key
        self.timeout = settings.api_timeout

    async def search_people(
        self,
        domain: str,
        titles: List[str],
        location: str = "Germany"
    ) -> List[DecisionMaker]:
        """
        Search for decision makers at a company.
        Uses mixed_people/search endpoint (FREE - no credits consumed).
        """
        if not self.api_key:
            logger.warning("Apollo API key not configured")
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Build query params for titles
            params = {
                "per_page": 10
            }

            # Build URL with array params
            url = f"{APOLLO_BASE_URL}/mixed_people/search"

            # Request body
            body = {
                "q_organization_domains": domain,
                "person_titles": titles,
                "person_locations": [location],
                "per_page": 10
            }

            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "x-api-key": self.api_key
            }

            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

                people = data.get("people", [])
                logger.info(f"Apollo found {len(people)} people at {domain}")

                results = []
                for person in people:
                    dm = DecisionMaker(
                        name=f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                        first_name=person.get("first_name"),
                        last_name=person.get("last_name"),
                        title=person.get("title"),
                        linkedin_url=person.get("linkedin_url"),
                        email=person.get("email"),  # May be None without enrichment
                        apollo_id=person.get("id")
                    )
                    results.append(dm)

                return results

            except httpx.HTTPStatusError as e:
                logger.error(f"Apollo API error: {e.response.status_code} - {e.response.text}")
                return []
            except Exception as e:
                logger.error(f"Apollo request failed: {e}")
                return []

    async def search_organization(self, company_name: str) -> Optional[CompanyInfo]:
        """
        Search for company information.
        Returns company details including domain, industry, etc.
        """
        if not self.api_key:
            logger.warning("Apollo API key not configured")
            return None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{APOLLO_BASE_URL}/mixed_companies/search"

            body = {
                "q_organization_name": company_name,
                "per_page": 3
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key
            }

            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

                organizations = data.get("organizations", [])
                if not organizations:
                    logger.info(f"No organization found for: {company_name}")
                    return None

                # Take best match
                org = organizations[0]

                return CompanyInfo(
                    name=org.get("name", company_name),
                    domain=org.get("primary_domain") or org.get("website_url", "").replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0],
                    industry=org.get("industry"),
                    employee_count=org.get("estimated_num_employees"),
                    location=self._format_location(org),
                    phone=org.get("phone"),
                    website=org.get("website_url"),
                    linkedin_url=org.get("linkedin_url")
                )

            except httpx.HTTPStatusError as e:
                logger.error(f"Apollo org search error: {e.response.status_code}")
                return None
            except Exception as e:
                logger.error(f"Apollo org search failed: {e}")
                return None

    async def enrich_person(self, person_id: str) -> Optional[DecisionMaker]:
        """
        Enrich a person to get email/phone (COSTS CREDITS).
        Only use when needed and Kaspr/FullEnrich fail.
        """
        if not self.api_key:
            return None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{APOLLO_BASE_URL}/people/match"

            body = {
                "id": person_id,
                "reveal_personal_emails": True,
                "reveal_phone_number": True
            }

            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key
            }

            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

                person = data.get("person", {})
                if not person:
                    return None

                return DecisionMaker(
                    name=f"{person.get('first_name', '')} {person.get('last_name', '')}".strip(),
                    first_name=person.get("first_name"),
                    last_name=person.get("last_name"),
                    title=person.get("title"),
                    linkedin_url=person.get("linkedin_url"),
                    email=person.get("email"),
                    apollo_id=person.get("id")
                )

            except Exception as e:
                logger.error(f"Apollo enrichment failed: {e}")
                return None

    def _format_location(self, org: dict) -> Optional[str]:
        """Format organization location."""
        parts = []
        if org.get("city"):
            parts.append(org["city"])
        if org.get("state"):
            parts.append(org["state"])
        if org.get("country"):
            parts.append(org["country"])
        return ", ".join(parts) if parts else None
