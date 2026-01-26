"""
BetterContact API client for phone/email enrichment.

Waterfall enrichment service that tries 20+ providers.
Used as first enrichment option before FullEnrich and Kaspr.

API: https://app.bettercontact.rocks/api/v2
"""

import logging
import asyncio
import httpx
from typing import Optional, List
from dataclasses import dataclass

from config import get_settings
from models import PhoneResult, PhoneSource, PhoneType

logger = logging.getLogger(__name__)

BETTERCONTACT_BASE_URL = "https://app.bettercontact.rocks/api/v2"

# Singleton instance to persist state across calls
_bettercontact_instance = None


@dataclass
class BetterContactResult:
    """Result from BetterContact enrichment."""
    phones: List[PhoneResult]
    emails: List[str]
    success: bool = False


class BetterContactClient:
    """
    BetterContact API client for waterfall phone/email enrichment.

    Tries 20+ data providers in sequence to maximize hit rate.
    Costs: ~1 credit per successful enrichment
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.bettercontact_api_key
        self.timeout = settings.api_timeout
        self.max_poll_attempts = 20  # Max 40 seconds polling
        self.poll_interval = 2  # seconds
        self._api_disabled = False  # Track if API is unavailable
        self._credits_checked = False
        self._has_credits = True

    async def check_credits(self) -> bool:
        """Check if we have credits available."""
        if not self.api_key:
            return False

        if self._credits_checked:
            return self._has_credits

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{BETTERCONTACT_BASE_URL}/account"
                headers = {"X-API-Key": self.api_key}

                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                credits_left = data.get("credits_left", 0)
                self._credits_checked = True
                self._has_credits = credits_left > 0

                logger.info(f"BetterContact credits: {credits_left}")

                if not self._has_credits:
                    self._api_disabled = True
                    logger.warning("BetterContact: No credits available")

                return self._has_credits

        except Exception as e:
            logger.warning(f"BetterContact credit check failed: {e}")
            self._credits_checked = True
            self._has_credits = False
            return False

    async def enrich(
        self,
        first_name: str,
        last_name: str,
        company_name: Optional[str] = None,
        domain: Optional[str] = None,
        linkedin_url: Optional[str] = None
    ) -> Optional[BetterContactResult]:
        """
        Enrich contact to get phone and email.

        Args:
            first_name: First name
            last_name: Last name
            company_name: Company name
            domain: Company domain
            linkedin_url: LinkedIn profile URL (optional)

        Returns:
            BetterContactResult with phones and emails, or None
        """
        if not self.api_key:
            logger.warning("BetterContact API key not configured")
            return None

        if self._api_disabled:
            logger.warning("BetterContact API disabled (no credits)")
            return None

        # Check credits on first call
        if not self._credits_checked:
            has_credits = await self.check_credits()
            if not has_credits:
                return None

        if not company_name and not domain:
            logger.warning("BetterContact requires company_name or domain")
            return None

        # Start enrichment
        request_id = await self._start_enrichment(
            first_name, last_name, company_name, domain, linkedin_url
        )

        if not request_id:
            return None

        # Poll for results
        return await self._poll_results(request_id)

    async def _start_enrichment(
        self,
        first_name: str,
        last_name: str,
        company_name: Optional[str],
        domain: Optional[str],
        linkedin_url: Optional[str]
    ) -> Optional[str]:
        """Start enrichment and return request_id."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{BETTERCONTACT_BASE_URL}/async"

            # Build lead data
            lead = {
                "first_name": first_name,
                "last_name": last_name,
            }

            if company_name:
                lead["company"] = company_name
            if domain:
                lead["company_domain"] = domain
            if linkedin_url:
                lead["linkedin_url"] = linkedin_url

            body = {
                "data": [lead],
                "enrich_email_address": True,
                "enrich_phone_number": True
            }

            headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }

            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

                request_id = data.get("id")
                logger.info(f"BetterContact started: {request_id}")
                return request_id

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"BetterContact start error: {status_code} - {e.response.text}")

                # Handle payment/credit issues
                if status_code in (402, 403, 429):
                    self._api_disabled = True
                    logger.error(f"BetterContact API disabled (status {status_code})")

                return None
            except Exception as e:
                logger.error(f"BetterContact start failed: {e}")
                return None

    async def _poll_results(self, request_id: str) -> Optional[BetterContactResult]:
        """Poll for enrichment results."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{BETTERCONTACT_BASE_URL}/async/{request_id}"

            headers = {"X-API-Key": self.api_key}

            for attempt in range(self.max_poll_attempts):
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    status = data.get("status", "").lower()
                    logger.info(f"BetterContact status: {status} (attempt {attempt + 1}/{self.max_poll_attempts})")

                    if status == "terminated":
                        return self._parse_results(data)
                    elif status in ["failed", "error"]:
                        logger.warning(f"BetterContact enrichment {status}")
                        return None

                    # Still processing, wait and retry
                    await asyncio.sleep(self.poll_interval)

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    logger.error(f"BetterContact poll HTTP error: {status_code}")

                    # Immediately stop on payment/auth errors
                    if status_code in (402, 403, 401):
                        logger.error(f"BetterContact API error {status_code} - stopping")
                        self._api_disabled = True
                        return None

                    if attempt >= 3:
                        logger.error(f"BetterContact HTTP error persists, giving up")
                        return None

                    await asyncio.sleep(self.poll_interval)

                except Exception as e:
                    logger.error(f"BetterContact poll error: {e}")
                    if attempt >= 3:
                        return None
                    await asyncio.sleep(self.poll_interval)

            logger.warning(f"BetterContact polling timeout for {request_id}")
            return None

    def _parse_results(self, data: dict) -> BetterContactResult:
        """Parse BetterContact response into structured result."""
        phones = []
        emails = []

        items = data.get("data", [])
        logger.info(f"BetterContact parsing {len(items)} items")

        for item in items:
            # Check if enrichment succeeded
            if not item.get("enriched"):
                continue

            # Extract email
            email = item.get("contact_email_address")
            email_status = item.get("contact_email_address_status", "")

            # Only accept deliverable or catch_all_safe emails
            if email and email_status in ["deliverable", "catch_all_safe"]:
                emails.append(email)

            # Extract phone number - try various field names
            phone_fields = [
                "contact_phone_number",
                "contact_mobile_phone",
                "contact_phone",
                "phone_number",
                "mobile_phone",
                "phone"
            ]

            for field in phone_fields:
                phone = item.get(field)
                if phone:
                    phone_type = self._determine_phone_type(phone)
                    phones.append(PhoneResult(
                        number=phone,
                        type=phone_type,
                        source=PhoneSource.BETTERCONTACT
                    ))
                    break

        # Remove duplicates
        emails = list(set(e for e in emails if e))

        success = len(phones) > 0 or len(emails) > 0
        logger.info(f"BetterContact result: {len(phones)} phones, {len(emails)} emails")

        return BetterContactResult(
            phones=phones,
            emails=emails,
            success=success
        )

    def _determine_phone_type(self, number: str) -> PhoneType:
        """Determine if phone is mobile or landline."""
        import re
        clean = re.sub(r'[^\d+]', '', number)

        # German mobile: +49 15x, +49 16x, +49 17x
        if re.match(r'(\+49|0049|49)?1[567]\d', clean):
            return PhoneType.MOBILE
        # Austrian mobile: +43 6xx
        if re.match(r'(\+43|0043|43)?6\d', clean):
            return PhoneType.MOBILE
        # Swiss mobile: +41 7x
        if re.match(r'(\+41|0041|41)?7[6789]\d', clean):
            return PhoneType.MOBILE

        return PhoneType.UNKNOWN


def get_bettercontact_client() -> BetterContactClient:
    """Get singleton BetterContact client instance."""
    global _bettercontact_instance
    if _bettercontact_instance is None:
        _bettercontact_instance = BetterContactClient()
    return _bettercontact_instance
