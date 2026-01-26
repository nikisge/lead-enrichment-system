import logging
import re
import httpx
from typing import Optional, List
from dataclasses import dataclass

from config import get_settings
from models import PhoneResult, PhoneSource, PhoneType

logger = logging.getLogger(__name__)

KASPR_BASE_URL = "https://api.developers.kaspr.io"

@dataclass
class KasprResult:
    """Result from Kaspr enrichment."""
    phones: List[PhoneResult]
    emails: List[str]
    success: bool


class KasprClient:
    """
    Kaspr API client for phone enrichment via LinkedIn.

    Note: Each request creates a fresh client to avoid global state issues.
    API availability is checked per-request, not cached globally.
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.kaspr_api_key
        self.timeout = settings.api_timeout

    async def enrich_by_linkedin(
        self,
        linkedin_url: str,
        name: str
    ) -> Optional[KasprResult]:
        """
        Enrich contact using LinkedIn URL.
        Costs 1 credit per request.

        Args:
            linkedin_url: Standard LinkedIn profile URL (not SalesNavigator)
            name: Full name of the person
        """
        if not self.api_key:
            logger.warning("Kaspr API key not configured")
            return None

        # Extract LinkedIn ID from URL
        linkedin_id = self._extract_linkedin_id(linkedin_url)
        if not linkedin_id:
            logger.warning(f"Could not extract LinkedIn ID from: {linkedin_url}")
            return None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = f"{KASPR_BASE_URL}/profile/linkedin"

            body = {
                "name": name,
                "id": linkedin_id
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "accept-version": "v2.0"
            }

            try:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()

                phones = []
                emails = []

                # Kaspr API returns data in "profile" object
                profile = data.get("profile", data)

                # Extract phones from profile
                phone_data = profile.get("phones", [])
                if isinstance(phone_data, list):
                    for phone in phone_data:
                        if isinstance(phone, dict):
                            number = phone.get("phoneNumber") or phone.get("phone")
                            phone_type = self._determine_phone_type(
                                phone.get("phoneType", ""),
                                number
                            )
                        else:
                            number = str(phone)
                            phone_type = self._determine_phone_type("", number)

                        if number:
                            phones.append(PhoneResult(
                                number=number,
                                type=phone_type,
                                source=PhoneSource.KASPR
                            ))

                # Also check for starryPhone field (single best phone)
                if not phones and profile.get("starryPhone"):
                    starry = profile["starryPhone"]
                    if isinstance(starry, str):
                        phones.append(PhoneResult(
                            number=starry,
                            type=self._determine_phone_type("", starry),
                            source=PhoneSource.KASPR
                        ))

                # Extract emails from profile
                # Check starryWorkEmail (best work email)
                if profile.get("starryWorkEmail"):
                    emails.append(profile["starryWorkEmail"])
                # Check starryDirectEmail (best personal email)
                if profile.get("starryDirectEmail"):
                    emails.append(profile["starryDirectEmail"])
                # Check workEmails array
                work_emails = profile.get("workEmails", [])
                if isinstance(work_emails, list):
                    emails.extend(work_emails)
                # Check directEmails array
                direct_emails = profile.get("directEmails", [])
                if isinstance(direct_emails, list):
                    emails.extend(direct_emails)

                # Also check emails array (with email objects)
                email_data = profile.get("emails", [])
                if isinstance(email_data, list):
                    for email in email_data:
                        if isinstance(email, dict):
                            emails.append(email.get("email", ""))
                        else:
                            emails.append(str(email))

                # Remove duplicates and empty strings
                emails = list(set(e for e in emails if e))

                success = len(phones) > 0 or len(emails) > 0
                logger.info(f"Kaspr enrichment: {len(phones)} phones, {len(emails)} emails")

                return KasprResult(
                    phones=phones,
                    emails=emails,
                    success=success
                )

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                logger.error(f"Kaspr API error: {status_code} - {e.response.text}")

                # Log payment/auth errors but don't disable globally
                # Each request should try fresh
                if status_code in (402, 403, 401):
                    logger.error(f"Kaspr API auth/payment error (status {status_code}) - check billing")

                return None
            except Exception as e:
                logger.error(f"Kaspr request failed: {e}")
                return None

    def _extract_linkedin_id(self, url: str) -> Optional[str]:
        """Extract LinkedIn profile ID from URL."""
        if not url:
            return None

        # Clean URL
        url = url.strip().rstrip("/")

        # Pattern: linkedin.com/in/profile-id
        patterns = [
            r'linkedin\.com/in/([^/?]+)',
            r'linkedin\.com/pub/([^/?]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _determine_phone_type(self, type_str: str, number: str) -> PhoneType:
        """Determine if phone is mobile or landline."""
        type_lower = type_str.lower() if type_str else ""

        if "mobile" in type_lower or "cell" in type_lower:
            return PhoneType.MOBILE
        if "landline" in type_lower or "work" in type_lower or "office" in type_lower:
            return PhoneType.LANDLINE

        # Check German mobile prefixes
        if number:
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


def get_kaspr_client() -> KasprClient:
    """
    Get a fresh Kaspr client instance.

    Creates new instance per request to avoid global state issues
    when handling concurrent requests.
    """
    return KasprClient()
