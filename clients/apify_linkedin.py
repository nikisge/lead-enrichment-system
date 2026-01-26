"""
Apify LinkedIn Profile Scraper client for employment verification.

Uses the "supreme_coder/linkedin-profile-scraper" actor (No Cookies required).
Cost: ~$3 per 1,000 profiles = $0.003 per profile

Purpose: Verify if a person currently works at a specific company
by checking their LinkedIn employment history.
"""

import logging
import asyncio
import httpx
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime

from config import get_settings

logger = logging.getLogger(__name__)

APIFY_BASE_URL = "https://api.apify.com/v2"
# Actor: supreme_coder/linkedin-profile-scraper (No cookies needed)
# Note: In API calls, use ~ instead of / for actor names
LINKEDIN_ACTOR_ID = "supreme_coder~linkedin-profile-scraper"

# Singleton instance
_apify_linkedin_instance = None


@dataclass
class LinkedInExperience:
    """A single work experience entry from LinkedIn."""
    company_name: str
    title: str
    start_date: Optional[str] = None  # e.g., "Jan 2022"
    end_date: Optional[str] = None    # None = "Present" = currently employed
    is_current: bool = False
    location: Optional[str] = None


@dataclass
class LinkedInProfile:
    """Parsed LinkedIn profile data."""
    name: str
    headline: Optional[str] = None
    location: Optional[str] = None
    experiences: List[LinkedInExperience] = None
    current_company: Optional[str] = None
    current_title: Optional[str] = None
    profile_url: str = ""

    def __post_init__(self):
        if self.experiences is None:
            self.experiences = []


@dataclass
class EmploymentVerification:
    """Result of employment verification."""
    is_currently_employed: bool
    company_name_matched: str = ""  # The company name we matched
    current_title: str = ""
    confidence: str = "low"  # low, medium, high
    verification_note: str = ""
    profile: Optional[LinkedInProfile] = None


class ApifyLinkedInClient:
    """
    Apify LinkedIn Profile Scraper client.

    Uses the no-cookies actor to scrape LinkedIn profiles
    and verify current employment.
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.apify_api_key if hasattr(settings, 'apify_api_key') else ""
        self.timeout = settings.api_timeout
        self._api_disabled = False

    async def verify_employment(
        self,
        linkedin_url: str,
        expected_company: str
    ) -> EmploymentVerification:
        """
        Verify if a person currently works at the expected company.

        Args:
            linkedin_url: LinkedIn profile URL
            expected_company: Company name to verify against

        Returns:
            EmploymentVerification with result
        """
        if not self.api_key:
            logger.warning("Apify API key not configured")
            return EmploymentVerification(
                is_currently_employed=True,  # Assume true if can't verify
                confidence="low",
                verification_note="Apify API key nicht konfiguriert - keine Verifizierung möglich"
            )

        if self._api_disabled:
            logger.warning("Apify API disabled")
            return EmploymentVerification(
                is_currently_employed=True,
                confidence="low",
                verification_note="Apify API deaktiviert"
            )

        # Scrape the LinkedIn profile
        profile = await self.scrape_profile(linkedin_url)

        if not profile:
            return EmploymentVerification(
                is_currently_employed=True,  # Assume true if can't scrape
                confidence="low",
                verification_note="LinkedIn Profil konnte nicht gescraped werden"
            )

        # Check if currently employed at expected company
        return self._verify_against_company(profile, expected_company)

    async def scrape_profile(self, linkedin_url: str) -> Optional[LinkedInProfile]:
        """
        Scrape a LinkedIn profile using Apify.

        Args:
            linkedin_url: LinkedIn profile URL

        Returns:
            LinkedInProfile or None if failed
        """
        if not self.api_key:
            return None

        try:
            # Start the actor run
            run_id = await self._start_actor_run(linkedin_url)
            if not run_id:
                return None

            # Wait for completion and get results
            return await self._get_run_results(run_id)

        except Exception as e:
            logger.error(f"Apify scrape failed: {e}")
            return None

    async def _start_actor_run(self, linkedin_url: str) -> Optional[str]:
        """Start an Apify actor run for the given LinkedIn URL."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Use the sync run endpoint (waits for completion)
            url = f"{APIFY_BASE_URL}/acts/{LINKEDIN_ACTOR_ID}/runs"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # Actor input - URLs must be objects with 'url' field
            run_input = {
                "urls": [{"url": linkedin_url}]
            }

            try:
                response = await client.post(
                    url,
                    json=run_input,
                    headers=headers,
                    timeout=60.0  # Longer timeout for actor start
                )
                response.raise_for_status()
                data = response.json()

                run_id = data.get("data", {}).get("id")
                logger.info(f"Apify actor started: {run_id}")
                return run_id

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"Apify start error: {status_code} - {e.response.text}")

                if status_code in (401, 402, 403):
                    self._api_disabled = True
                    logger.error("Apify API disabled (auth/payment issue)")

                return None
            except Exception as e:
                logger.error(f"Apify start failed: {e}")
                return None

    async def _get_run_results(self, run_id: str) -> Optional[LinkedInProfile]:
        """Poll for actor run completion and get results."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            # Poll for run completion
            max_attempts = 30  # 60 seconds max
            for attempt in range(max_attempts):
                try:
                    # Check run status
                    status_url = f"{APIFY_BASE_URL}/actor-runs/{run_id}"
                    response = await client.get(status_url, headers=headers)
                    response.raise_for_status()

                    run_data = response.json().get("data", {})
                    status = run_data.get("status")

                    logger.info(f"Apify run status: {status} (attempt {attempt + 1})")

                    if status == "SUCCEEDED":
                        # Get dataset items
                        dataset_id = run_data.get("defaultDatasetId")
                        return await self._fetch_dataset(dataset_id)

                    elif status in ("FAILED", "ABORTED", "TIMED-OUT"):
                        logger.error(f"Apify run {status}")
                        return None

                    # Still running, wait
                    await asyncio.sleep(2)

                except Exception as e:
                    logger.error(f"Apify poll error: {e}")
                    if attempt >= 5:
                        return None
                    await asyncio.sleep(2)

            logger.warning("Apify run timeout")
            return None

    async def _fetch_dataset(self, dataset_id: str) -> Optional[LinkedInProfile]:
        """Fetch results from Apify dataset."""
        if not dataset_id:
            return None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {"Authorization": f"Bearer {self.api_key}"}

            url = f"{APIFY_BASE_URL}/datasets/{dataset_id}/items"

            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()

                items = response.json()
                if not items or len(items) == 0:
                    logger.warning("Apify returned empty dataset")
                    return None

                # Parse the first profile
                return self._parse_profile(items[0])

            except Exception as e:
                logger.error(f"Apify dataset fetch failed: {e}")
                return None

    def _parse_profile(self, data: dict) -> LinkedInProfile:
        """Parse Apify response into LinkedInProfile."""
        experiences = []

        # Parse positions from the supreme_coder actor format
        positions = data.get("positions") or data.get("experiences") or []

        for pos in positions:
            # Get company name from nested company object or direct field
            company_obj = pos.get("company", {})
            company = (
                company_obj.get("name") if isinstance(company_obj, dict) else
                pos.get("companyName") or
                pos.get("company") or
                ""
            )

            title = pos.get("title") or pos.get("position") or ""

            # Check time period for current status
            time_period = pos.get("timePeriod", {})
            end_date_obj = time_period.get("endDate")
            start_date_obj = time_period.get("startDate")

            # endDate is None for current positions
            is_current = end_date_obj is None

            # Format dates as strings
            start_date = None
            if start_date_obj:
                year = start_date_obj.get("year", "")
                month = start_date_obj.get("month", "")
                start_date = f"{month}/{year}" if month else str(year)

            end_date = None
            if end_date_obj:
                year = end_date_obj.get("year", "")
                month = end_date_obj.get("month", "")
                end_date = f"{month}/{year}" if month else str(year)

            experiences.append(LinkedInExperience(
                company_name=company,
                title=title,
                start_date=start_date,
                end_date=end_date,
                is_current=is_current,
                location=pos.get("locationName") or pos.get("location")
            ))

        # Use direct fields for current company (faster than iterating)
        current_company = data.get("companyName")
        current_title = data.get("jobTitle")

        # Fallback to first current position if direct fields missing
        if not current_company:
            for exp in experiences:
                if exp.is_current:
                    current_company = exp.company_name
                    current_title = exp.title
                    break

        # Build full name
        name = data.get("fullName") or f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()

        return LinkedInProfile(
            name=name,
            headline=data.get("headline"),
            location=data.get("geoLocationName") or data.get("location"),
            experiences=experiences,
            current_company=current_company,
            current_title=current_title,
            profile_url=data.get("inputUrl") or data.get("url") or ""
        )

    def _verify_against_company(
        self,
        profile: LinkedInProfile,
        expected_company: str
    ) -> EmploymentVerification:
        """Check if profile shows current employment at expected company."""
        expected_lower = expected_company.lower().strip()

        # Remove common suffixes for matching
        expected_normalized = self._normalize_company_name(expected_lower)

        # Check current experiences
        for exp in profile.experiences:
            if not exp.is_current:
                continue

            company_normalized = self._normalize_company_name(exp.company_name.lower())

            # Check for match
            if self._company_names_match(expected_normalized, company_normalized):
                return EmploymentVerification(
                    is_currently_employed=True,
                    company_name_matched=exp.company_name,
                    current_title=exp.title,
                    confidence="high",
                    verification_note=f"Aktuell bei {exp.company_name} als {exp.title}",
                    profile=profile
                )

        # No current match found - check if they have ANY current job
        current_jobs = [e for e in profile.experiences if e.is_current]

        if current_jobs:
            # They work somewhere else
            other_company = current_jobs[0].company_name
            return EmploymentVerification(
                is_currently_employed=False,
                company_name_matched="",
                current_title=current_jobs[0].title,
                confidence="high",
                verification_note=f"Arbeitet aktuell bei {other_company}, NICHT bei {expected_company}",
                profile=profile
            )
        else:
            # No current job listed - might be between jobs or profile not updated
            return EmploymentVerification(
                is_currently_employed=False,
                company_name_matched="",
                current_title="",
                confidence="medium",
                verification_note=f"Keine aktuelle Position auf LinkedIn - möglicherweise veraltet",
                profile=profile
            )

    def _normalize_company_name(self, name: str) -> str:
        """Normalize company name for comparison."""
        # Remove common suffixes
        suffixes = [
            " gmbh", " ag", " se", " kg", " ohg", " gbr", " ug",
            " gmbh & co. kg", " gmbh & co kg", " co. kg",
            " inc", " inc.", " ltd", " ltd.", " llc", " corp", " corp.",
            " holding", " group", " international"
        ]

        normalized = name.lower().strip()
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()

        return normalized

    def _company_names_match(self, expected: str, actual: str) -> bool:
        """Check if company names match (fuzzy)."""
        # Exact match
        if expected == actual:
            return True

        # One contains the other
        if expected in actual or actual in expected:
            return True

        # Check word overlap
        expected_words = set(expected.split())
        actual_words = set(actual.split())

        # If main word matches (usually first word)
        if expected_words and actual_words:
            # Check if significant words overlap
            common = expected_words & actual_words
            if len(common) >= 1 and len(common) >= len(expected_words) * 0.5:
                return True

        return False


def get_apify_linkedin_client() -> ApifyLinkedInClient:
    """Get singleton Apify LinkedIn client instance."""
    global _apify_linkedin_instance
    if _apify_linkedin_instance is None:
        _apify_linkedin_instance = ApifyLinkedInClient()
    return _apify_linkedin_instance
