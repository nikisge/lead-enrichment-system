"""
Apify LinkedIn Profile Scraper client for employment verification.

Uses the "harvestapi/linkedin-profile-scraper" actor (No Cookies required).
Cost: ~$4 per 1,000 profiles = $0.004 per profile

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
# Actor: harvestapi/linkedin-profile-scraper (No cookies needed)
# Note: In API calls, use ~ instead of / for actor names
LINKEDIN_ACTOR_ID = "harvestapi~linkedin-profile-scraper"

@dataclass
class LinkedInExperience:
    """A single work experience entry from LinkedIn."""
    company_name: str
    title: str
    start_date: Optional[str] = None  # e.g., "Jan 2024"
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
    Apify LinkedIn Profile Scraper client (HarvestAPI).

    Uses the no-cookies actor to scrape LinkedIn profiles
    and verify current employment.

    Note: Each request creates a fresh client to avoid global state issues.
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.apify_api_key if hasattr(settings, 'apify_api_key') else ""
        self.timeout = settings.api_timeout

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
                is_currently_employed=False,  # Can't verify = not verified!
                confidence="none",
                verification_note="Apify API key nicht konfiguriert - keine Verifizierung möglich"
            )

        try:
            # Scrape the LinkedIn profile
            profile = await self.scrape_profile(linkedin_url)

            if not profile:
                logger.warning("Could not scrape LinkedIn profile - treating as NOT verified")
                return EmploymentVerification(
                    is_currently_employed=False,  # Can't verify = not verified!
                    confidence="none",
                    verification_note="LinkedIn Profil konnte nicht gescraped werden - nicht verifiziert"
                )

            # Check if currently employed at expected company
            return self._verify_against_company(profile, expected_company)

        except Exception as e:
            logger.error(f"Employment verification failed unexpectedly: {e}")
            return EmploymentVerification(
                is_currently_employed=False,
                confidence="none",
                verification_note=f"Verifizierung fehlgeschlagen: {str(e)[:100]}"
            )

    async def scrape_profile(self, linkedin_url: str) -> Optional[LinkedInProfile]:
        """
        Scrape a LinkedIn profile using Apify (HarvestAPI actor).

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
            url = f"{APIFY_BASE_URL}/acts/{LINKEDIN_ACTOR_ID}/runs"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            # HarvestAPI input format: use "queries" field with URLs
            run_input = {
                "queries": [linkedin_url]
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
                logger.info(f"Apify HarvestAPI actor started: {run_id}")
                return run_id

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"Apify start error: {status_code} - {e.response.text[:200]}")

                if status_code in (401, 402, 403):
                    logger.error("Apify API auth/payment error - check billing or API key")

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

                item = items[0]

                # Check for API error response (profile not accessible)
                if item.get("error") or item.get("status") in (403, 404):
                    error_details = item.get("error", [])
                    error_msg = error_details[0].get("error", "Unknown error") if error_details and isinstance(error_details, list) else str(error_details)
                    logger.warning(f"Apify profile not accessible: {error_msg}")
                    return None

                # Parse the first profile
                return self._parse_profile(item)

            except Exception as e:
                logger.error(f"Apify dataset fetch failed: {e}")
                return None

    def _parse_profile(self, data: dict) -> Optional[LinkedInProfile]:
        """
        Parse HarvestAPI response into LinkedInProfile.

        HarvestAPI format:
        - firstName, lastName (no fullName)
        - currentPosition: [{companyName, dateRange: {start, end}}]
        - experience: [{position, companyName, startDate, endDate, ...}]
        - headline, location.linkedinText, linkedinUrl
        """
        try:
            experiences = []

            # --- Extract current companies from "currentPosition" (most reliable!) ---
            current_positions = data.get("currentPosition") or []
            current_company_names_from_api = []
            current_company_from_api = None  # Primary (first) company
            current_title_from_api = None

            if current_positions and isinstance(current_positions, list):
                for cp in current_positions:
                    cp_name = cp.get("companyName")
                    if cp_name:
                        current_company_names_from_api.append(cp_name)
                if current_company_names_from_api:
                    current_company_from_api = current_company_names_from_api[0]

            # Also try headline as title hint (e.g. "Staff Pharmacist at CVS Health")
            headline = data.get("headline") or ""

            logger.info(f"HarvestAPI direct fields: currentCompanies={current_company_names_from_api}, headline='{headline[:80]}'")

            # --- Parse experience array ---
            experience_list = data.get("experience") or []
            logger.info(f"Parsing {len(experience_list)} experience entries from LinkedIn profile")

            for idx, exp in enumerate(experience_list):
                company = exp.get("companyName") or ""
                # HarvestAPI uses "position" field for job title
                title = exp.get("position") or exp.get("title") or ""

                # Parse dates from HarvestAPI format
                start_date_obj = exp.get("startDate")  # {month: "Jan", year: 2024, text: "Jan 2024"}
                end_date_obj = exp.get("endDate")       # {text: "Present"} or {month: "Dec", year: 2023, text: "Dec 2023"}

                start_date = None
                end_date = None
                is_current = False

                if start_date_obj and isinstance(start_date_obj, dict):
                    start_date = start_date_obj.get("text") or self._format_date_obj(start_date_obj)

                if end_date_obj and isinstance(end_date_obj, dict):
                    end_text = end_date_obj.get("text", "")
                    if end_text.lower() in ("present", "heute", "current", "jetzt"):
                        # Currently employed here
                        is_current = True
                        end_date = None
                    else:
                        end_date = end_text or self._format_date_obj(end_date_obj)
                elif end_date_obj is None and start_date_obj is not None:
                    # No endDate at all = likely current position
                    is_current = True

                # Note: We trust endDate="Present" from the experience array as-is.
                # The currentPosition API field only shows the primary/featured position,
                # not all current positions. A person can legitimately work at multiple
                # companies simultaneously (e.g., consultant + full-time, co-chair + founder).
                # We do NOT override is_current based on currentPosition cross-check.

                # Extract title for current position
                if is_current and not current_title_from_api:
                    current_title_from_api = title

                logger.info(f"  Position {idx}: {company} | {title} | {start_date} - {end_date or 'Present'} | is_current={is_current}")

                experiences.append(LinkedInExperience(
                    company_name=company,
                    title=title,
                    start_date=start_date,
                    end_date=end_date,
                    is_current=is_current,
                    location=exp.get("location") or ""
                ))

            # --- Build profile ---
            name = f"{data.get('firstName', '')} {data.get('lastName', '')}".strip()
            if not name:
                name = data.get("fullName") or "Unknown"

            # Location from HarvestAPI: location.linkedinText or fallback
            location_obj = data.get("location")
            location = None
            if isinstance(location_obj, dict):
                location = location_obj.get("linkedinText") or location_obj.get("parsed", {}).get("text")
            elif isinstance(location_obj, str):
                location = location_obj

            logger.info(f"Profile parsed: {name} | Current: {current_company_from_api} as {current_title_from_api}")

            return LinkedInProfile(
                name=name,
                headline=headline,
                location=location,
                experiences=experiences,
                current_company=current_company_from_api,
                current_title=current_title_from_api,
                profile_url=data.get("linkedinUrl") or data.get("url") or ""
            )

        except Exception as e:
            logger.error(f"Failed to parse HarvestAPI profile data: {e}")
            return None

    def _format_date_obj(self, date_obj: dict) -> Optional[str]:
        """Format a HarvestAPI date object like {month: "Jan", year: 2024} to string."""
        month = date_obj.get("month", "")
        year = date_obj.get("year", "")
        if month and year:
            return f"{month} {year}"
        elif year:
            return str(year)
        return None

    def _verify_against_company(
        self,
        profile: LinkedInProfile,
        expected_company: str
    ) -> EmploymentVerification:
        """
        Check if profile shows current employment at expected company.

        Handles multiple current jobs: If person works at Company A + Company B,
        and we're looking for Company A, they ARE currently employed there.
        """
        expected_lower = expected_company.lower().strip()
        expected_normalized = self._normalize_company_name(expected_lower)

        logger.info(f"Verifying employment: {profile.name} at '{expected_company}'")
        logger.info(f"  Profile's primary company (from API): '{profile.current_company}'")

        # Collect ALL current jobs (person might have multiple!)
        current_jobs = [e for e in profile.experiences if e.is_current]

        # Also add the primary company from API if not already in list
        if profile.current_company:
            primary_in_list = any(
                self._company_names_match(
                    self._normalize_company_name(e.company_name.lower()),
                    self._normalize_company_name(profile.current_company.lower())
                )
                for e in current_jobs
            )
            if not primary_in_list:
                # Add primary company as a current job
                current_jobs.insert(0, LinkedInExperience(
                    company_name=profile.current_company,
                    title=profile.current_title or "",
                    is_current=True
                ))

        logger.info(f"  Total current positions: {len(current_jobs)}")
        for idx, job in enumerate(current_jobs):
            logger.info(f"    [{idx}] {job.company_name} | {job.title}")

        # Check if ANY current job matches the expected company
        for job in current_jobs:
            job_normalized = self._normalize_company_name(job.company_name.lower())
            logger.info(f"  Comparing: '{expected_normalized}' vs '{job_normalized}'")

            if self._company_names_match(expected_normalized, job_normalized):
                logger.info(f"  ✓ MATCH FOUND: {job.company_name}")
                return EmploymentVerification(
                    is_currently_employed=True,
                    company_name_matched=job.company_name,
                    current_title=job.title,
                    confidence="high",
                    verification_note=f"Aktuell bei {job.company_name} als {job.title}",
                    profile=profile
                )

        # No match found in any current job
        if current_jobs:
            other_companies = ", ".join([e.company_name for e in current_jobs[:3]])  # Max 3
            if len(current_jobs) > 3:
                other_companies += f" (+{len(current_jobs) - 3} weitere)"
            logger.warning(f"  ✗ NO MATCH: Person works at [{other_companies}], NOT at '{expected_company}'")
            return EmploymentVerification(
                is_currently_employed=False,
                company_name_matched="",
                current_title=current_jobs[0].title,
                confidence="high",
                verification_note=f"Arbeitet aktuell bei {other_companies}, NICHT bei {expected_company}",
                profile=profile
            )
        else:
            logger.warning(f"  ✗ NO CURRENT JOB: {profile.name} has no current position listed")
            return EmploymentVerification(
                is_currently_employed=False,
                company_name_matched="",
                current_title="",
                confidence="medium",
                verification_note=f"Keine aktuelle Position auf LinkedIn gefunden",
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
    """
    Get a fresh Apify LinkedIn client instance.

    Creates new instance per request to avoid global state issues
    when handling concurrent requests.
    """
    return ApifyLinkedInClient()
