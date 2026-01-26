import logging
import asyncio
import httpx
from typing import Optional, List
from dataclasses import dataclass

from config import get_settings
from models import PhoneResult, PhoneSource, PhoneType

logger = logging.getLogger(__name__)

FULLENRICH_BASE_URL = "https://app.fullenrich.com/api/v1"

# Singleton instance to persist state across calls
_fullenrich_instance = None


@dataclass
class FullEnrichResult:
    """Result from FullEnrich enrichment."""
    phones: List[PhoneResult]
    emails: List[str]
    linkedin_url: Optional[str] = None  # Can be extracted from social_medias
    success: bool = False


class FullEnrichClient:
    """
    FullEnrich API client for phone/email enrichment.
    Can work without LinkedIn URL (just name + company).
    Costs: Email = 1 credit, Mobile = 10 credits
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.fullenrich_api_key
        self.timeout = settings.api_timeout
        self.max_poll_attempts = 15  # Reduced from 30 - max 30 seconds polling
        self.poll_interval = 2  # seconds
        self._api_disabled = False  # Track if API is unavailable (e.g., no credits)

    async def enrich(
        self,
        first_name: str,
        last_name: str,
        company_name: Optional[str] = None,
        domain: Optional[str] = None,
        linkedin_url: Optional[str] = None
    ) -> Optional[FullEnrichResult]:
        """
        Enrich contact to get phone and email.
        Requires name + (company OR domain).
        LinkedIn URL is optional but improves hit rate.
        """
        if not self.api_key:
            logger.warning("FullEnrich API key not configured")
            return None

        if self._api_disabled:
            logger.warning("FullEnrich API disabled (no credits or rate limited)")
            return None

        if not company_name and not domain:
            logger.warning("FullEnrich requires company_name or domain")
            return None

        # Start enrichment
        enrichment_id = await self._start_enrichment(
            first_name, last_name, company_name, domain, linkedin_url
        )

        if not enrichment_id:
            return None

        # Poll for results (FullEnrich is async)
        return await self._poll_results(enrichment_id)

    async def _start_enrichment(
        self,
        first_name: str,
        last_name: str,
        company_name: Optional[str],
        domain: Optional[str],
        linkedin_url: Optional[str]
    ) -> Optional[str]:
        """Start bulk enrichment and return enrichment_id."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{FULLENRICH_BASE_URL}/contact/enrich/bulk"

            contact = {
                "firstname": first_name,
                "lastname": last_name,
                "enrich_fields": ["contact.emails", "contact.phones"]
            }

            if domain:
                contact["domain"] = domain
            if company_name:
                contact["company_name"] = company_name
            if linkedin_url:
                contact["linkedin_url"] = linkedin_url

            body = {
                "name": f"Enrichment {first_name} {last_name}",
                "datas": [contact]
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

                enrichment_id = data.get("enrichment_id")
                logger.info(f"FullEnrich started: {enrichment_id}")
                return enrichment_id

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"FullEnrich start error: {status_code} - {e.response.text}")

                # Handle payment/credit issues - disable API for this session
                if status_code in (402, 403, 429):
                    self._api_disabled = True
                    logger.error(f"FullEnrich API disabled for this session (status {status_code})")

                return None
            except Exception as e:
                logger.error(f"FullEnrich start failed: {e}")
                return None

    async def _poll_results(self, enrichment_id: str) -> Optional[FullEnrichResult]:
        """Poll for enrichment results."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{FULLENRICH_BASE_URL}/contact/enrich/bulk/{enrichment_id}"

            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            for attempt in range(self.max_poll_attempts):
                try:
                    response = await client.get(url, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    status = data.get("status", "").upper()
                    logger.info(f"FullEnrich status: {status} (attempt {attempt + 1}/{self.max_poll_attempts})")

                    if status == "FINISHED":
                        return self._parse_results(data)
                    elif status in ["CANCELED", "CREDITS_INSUFFICIENT", "RATE_LIMIT", "UNKNOWN"]:
                        logger.warning(f"FullEnrich enrichment {status}")
                        if status == "CREDITS_INSUFFICIENT":
                            self._api_disabled = True
                        return None
                    # CREATED, IN_PROGRESS → keep polling

                    # Still processing, wait and retry
                    await asyncio.sleep(self.poll_interval)

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    logger.error(f"FullEnrich poll HTTP error: {status_code}")

                    # Immediately stop on payment/auth errors
                    if status_code in (402, 403, 401):
                        logger.error(f"FullEnrich API error {status_code} - stopping immediately")
                        self._api_disabled = True
                        return None

                    # For rate limits, wait longer but eventually give up
                    if status_code == 429:
                        if attempt < 3:
                            logger.warning("FullEnrich rate limited, waiting 5s...")
                            await asyncio.sleep(5)
                            continue
                        else:
                            logger.error("FullEnrich rate limited too many times, giving up")
                            return None

                    # For other HTTP errors, don't retry endlessly
                    if attempt >= 3:
                        logger.error(f"FullEnrich HTTP error persists after {attempt + 1} attempts, giving up")
                        return None

                    await asyncio.sleep(self.poll_interval)

                except Exception as e:
                    logger.error(f"FullEnrich poll error: {e}")
                    # Don't keep polling on generic errors
                    if attempt >= 3:
                        logger.error(f"FullEnrich error persists after {attempt + 1} attempts, giving up")
                        return None
                    await asyncio.sleep(self.poll_interval)

            logger.warning(f"FullEnrich polling timeout for {enrichment_id}")
            return None

    def _parse_results(self, data: dict) -> FullEnrichResult:
        """Parse FullEnrich response into structured result."""
        phones = []
        emails = []

        items = data.get("datas", [])
        logger.info(f"FullEnrich parsing {len(items)} items")

        for item in items:
            # Contact data is nested inside "contact" key
            contact = item.get("contact", item)  # fallback to item if no contact key

            # Extract phones
            contact_phones = contact.get("phones", [])
            for phone in contact_phones:
                if isinstance(phone, dict):
                    number = phone.get("number") or phone.get("phone")
                    region = phone.get("region", "")
                    is_mobile = self._is_mobile_number(number) if number else False
                else:
                    number = str(phone)
                    is_mobile = self._is_mobile_number(number)

                if number:
                    # Add country code if region is DE and number doesn't have it
                    if region == "DE" and not number.startswith("+"):
                        number = "+49" + number.lstrip("0")

                    phones.append(PhoneResult(
                        number=number,
                        type=PhoneType.MOBILE if is_mobile else PhoneType.UNKNOWN,
                        source=PhoneSource.FULLENRICH
                    ))

            # Extract emails
            contact_emails = contact.get("emails", [])
            for email_item in contact_emails:
                if isinstance(email_item, dict):
                    email = email_item.get("email", "")
                    status = email_item.get("status", "")
                    # Only add valid/deliverable emails
                    if email and status not in ["INVALID"]:
                        emails.append(email)
                elif email_item:
                    emails.append(str(email_item))

            # Also check most_probable_email field
            if contact.get("most_probable_email"):
                emails.append(contact["most_probable_email"])

            # Direct fields as fallback
            if contact.get("email"):
                emails.append(contact["email"])
            if contact.get("phone"):
                phones.append(PhoneResult(
                    number=contact["phone"],
                    type=PhoneType.MOBILE if self._is_mobile_number(contact["phone"]) else PhoneType.UNKNOWN,
                    source=PhoneSource.FULLENRICH
                ))

        # Extract LinkedIn URL from social_medias
        linkedin_url = None
        for item in items:
            contact = item.get("contact", item)
            social_medias = contact.get("social_medias", [])
            for sm in social_medias:
                if isinstance(sm, dict):
                    sm_type = sm.get("type", "").lower()
                    sm_url = sm.get("url", "")
                    if sm_type == "linkedin" or "linkedin.com" in sm_url:
                        linkedin_url = sm_url
                        break
            if linkedin_url:
                break

        # Remove duplicates
        emails = list(set(e for e in emails if e))

        success = len(phones) > 0 or len(emails) > 0 or linkedin_url is not None
        logger.info(f"FullEnrich result: {len(phones)} phones, {len(emails)} emails, linkedin={linkedin_url}")

        return FullEnrichResult(
            phones=phones,
            emails=emails,
            linkedin_url=linkedin_url,
            success=success
        )

    def _is_mobile_number(self, number: str) -> bool:
        """Check if number is likely a mobile number."""
        import re
        clean = re.sub(r'[^\d+]', '', number)
        # German mobile: +49 15x, +49 16x, +49 17x
        if re.match(r'(\+49|0049|49)?1[567]\d', clean):
            return True
        # Austrian mobile: +43 6xx
        if re.match(r'(\+43|0043|43)?6\d', clean):
            return True
        # Swiss mobile: +41 7x
        if re.match(r'(\+41|0041|41)?7[6789]\d', clean):
            return True
        return False


def get_fullenrich_client() -> FullEnrichClient:
    """Get singleton FullEnrich client instance.

    This ensures the _api_disabled flag persists across calls,
    preventing repeated failed API calls when credits are exhausted.
    """
    global _fullenrich_instance
    if _fullenrich_instance is None:
        _fullenrich_instance = FullEnrichClient()
    return _fullenrich_instance
