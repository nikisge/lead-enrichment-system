import logging
import re
import httpx
from typing import Optional, List
from bs4 import BeautifulSoup
from dataclasses import dataclass

from config import get_settings
from models import PhoneResult, PhoneSource, PhoneType

logger = logging.getLogger(__name__)


@dataclass
class ImpressumResult:
    """Result from Impressum scraping."""
    phones: List[PhoneResult]
    emails: List[str]
    website_url: Optional[str] = None  # The actual URL we found
    address: Optional[str] = None  # Street address if found
    raw_text: Optional[str] = None  # Raw page text for AI extraction
    success: bool = False


@dataclass
class TeamMember:
    """A team member found on the company website."""
    name: str
    title: Optional[str] = None
    email: Optional[str] = None
    source_url: str = ""


@dataclass
class TeamPageResult:
    """Result from team page scraping."""
    members: List[TeamMember]
    source_url: Optional[str] = None
    success: bool = False


class ImpressumScraper:
    """
    Scrapes Impressum pages for contact information.
    Free fallback when paid services don't find a phone.
    """

    def __init__(self):
        settings = get_settings()
        self.google_api_key = settings.google_api_key
        self.google_cse_id = settings.google_cse_id
        self.timeout = settings.api_timeout

    async def scrape(
        self,
        company_name: str,
        domain: Optional[str] = None
    ) -> Optional[ImpressumResult]:
        """
        Find and scrape company Impressum page.

        Strategy:
        1. If domain known, try {domain}/impressum directly
        2. Use Google to find Impressum page
        3. Scrape found page for phone/email
        """
        urls_to_try = []

        # Try direct URLs first
        if domain:
            clean_domain = domain.replace("www.", "")
            urls_to_try.extend([
                f"https://www.{clean_domain}/impressum",
                f"https://{clean_domain}/impressum",
                f"https://www.{clean_domain}/impressum.html",
                f"https://{clean_domain}/de/impressum",
                f"https://www.{clean_domain}/kontakt",
            ])

        # Try to scrape each URL
        for url in urls_to_try:
            result = await self._scrape_url(url)
            if result and result.success:
                return result

        # Fallback: Google search
        if self.google_api_key and self.google_cse_id:
            google_url = await self._google_search(company_name, domain)
            if google_url:
                result = await self._scrape_url(google_url)
                if result and result.success:
                    return result

        return None

    async def scrape_team_page(
        self,
        company_name: str,
        domain: Optional[str] = None,
        job_category: Optional[str] = None
    ) -> Optional[TeamPageResult]:
        """
        Scrape company team/about page to find verified employees.
        These are VERIFIED contacts since they're on the company website.

        Args:
            company_name: Company name
            domain: Company domain
            job_category: Job category to prioritize relevant contacts

        Returns:
            TeamPageResult with list of team members
        """
        if not domain:
            logger.info("No domain for team page scraping")
            return None

        # Team page URLs to try
        clean_domain = domain.replace("www.", "")
        team_urls = [
            f"https://www.{clean_domain}/team",
            f"https://{clean_domain}/team",
            f"https://www.{clean_domain}/ueber-uns",
            f"https://{clean_domain}/ueber-uns",
            f"https://www.{clean_domain}/about",
            f"https://{clean_domain}/about",
            f"https://www.{clean_domain}/unternehmen",
            f"https://{clean_domain}/unternehmen",
            f"https://www.{clean_domain}/mitarbeiter",
            f"https://{clean_domain}/mitarbeiter",
            f"https://www.{clean_domain}/ansprechpartner",
            f"https://{clean_domain}/ansprechpartner",
            f"https://www.{clean_domain}/kontakt",
            f"https://{clean_domain}/kontakt",
        ]

        all_members = []

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        ) as client:
            # Try max 3 URLs to avoid too many requests
            tried = 0
            for url in team_urls:
                if tried >= 3:
                    break

                try:
                    response = await client.get(url)
                    if response.status_code != 200:
                        continue

                    tried += 1
                    soup = BeautifulSoup(response.text, "lxml")
                    members = self._extract_team_members(soup, url)

                    if members:
                        all_members.extend(members)
                        logger.info(f"Team page {url}: found {len(members)} members")

                except Exception as e:
                    logger.debug(f"Team page failed: {url} - {e}")
                    continue

        if not all_members:
            logger.info(f"No team members found for {company_name}")
            return None

        # Remove duplicates by name
        seen_names = set()
        unique_members = []
        for m in all_members:
            name_lower = m.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_members.append(m)

        # Prioritize HR/Recruiting contacts, then job-category relevant
        prioritized = self._prioritize_team_members(unique_members, job_category)

        logger.info(f"Team page scraping: {len(prioritized)} unique members for {company_name}")

        return TeamPageResult(
            members=prioritized[:5],  # Max 5 members
            source_url=team_urls[0] if all_members else None,
            success=len(prioritized) > 0
        )

    def _extract_team_members(self, soup: BeautifulSoup, source_url: str) -> List[TeamMember]:
        """Extract team member names and titles from HTML."""
        members = []

        # Common patterns for team member sections
        # Look for structured data (cards, divs with name+title)

        # Pattern 1: Look for elements with common class names
        team_containers = soup.find_all(['div', 'section', 'article'], class_=re.compile(
            r'team|member|employee|staff|person|profile|card', re.IGNORECASE
        ))

        for container in team_containers:
            name = None
            title = None

            # Try to find name in h2, h3, h4, strong, or class containing "name"
            name_elem = container.find(['h2', 'h3', 'h4', 'strong'], class_=re.compile(r'name', re.IGNORECASE))
            if not name_elem:
                name_elem = container.find(['h2', 'h3', 'h4', 'strong'])

            if name_elem:
                # Clean extracted text: normalize whitespace
                raw_text = name_elem.get_text(strip=True)
                name = ' '.join(raw_text.split())  # Collapse all whitespace

            # Try to find title in p, span, or class containing "title", "position", "role"
            title_elem = container.find(['p', 'span', 'div'], class_=re.compile(
                r'title|position|role|job|funktion', re.IGNORECASE
            ))
            if title_elem:
                raw_title = title_elem.get_text(strip=True)
                title = ' '.join(raw_title.split())  # Collapse all whitespace

            if name and self._is_valid_name(name):
                members.append(TeamMember(
                    name=name,
                    title=title,
                    source_url=source_url
                ))

        # Pattern 2: Look for text patterns like "Name - Title" or "Name, Title"
        if not members:
            text = soup.get_text(separator="\n")
            # Pattern: German names with titles
            patterns = [
                # "Max Müller - Geschäftsführer"
                r'([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)\s*[-–|]\s*([A-Za-zäöüÄÖÜß\s]+(?:leiter|manager|director|head|chef|führer|inhaber))',
                # "Geschäftsführer: Max Müller"
                r'(Geschäftsführer|CEO|Inhaber|Personalleiter|HR\s*Manager)[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches[:5]:  # Limit matches
                    if len(match) == 2:
                        name, title = match[0], match[1]
                        # Check which one is the name
                        if self._is_valid_name(name):
                            members.append(TeamMember(name=name, title=title, source_url=source_url))
                        elif self._is_valid_name(title):
                            members.append(TeamMember(name=title, title=name, source_url=source_url))

        return members

    def _is_valid_name(self, text: str) -> bool:
        """Check if text looks like a valid person name."""
        if not text or len(text) < 4 or len(text) > 50:
            return False

        # Reject if contains whitespace artifacts (tabs, multiple newlines)
        if '\t' in text or '\n\n' in text or '  ' in text:
            return False

        # Clean the text (single newlines -> spaces)
        text = ' '.join(text.split())

        # Must have at least 2 words (first + last name)
        words = text.split()
        if len(words) < 2:
            return False

        # Filter out obvious non-names
        invalid_patterns = [
            'kontakt', 'email', 'telefon', 'adresse', 'impressum',
            'gmbh', 'ag', 'kg', 'mbh', 'ohg', 'ug',
            'straße', 'str.', 'platz', 'weg',
            '@', 'www', 'http', '.de', '.com',
            'mehr erfahren', 'weiterlesen', 'zum profil'
        ]

        text_lower = text.lower()
        if any(p in text_lower for p in invalid_patterns):
            return False

        # Filter out job titles that are NOT person names
        job_title_patterns = [
            'präsident', 'vizepräsident', 'vize',
            'teamleiter', 'abteilungsleiter', 'bereichsleiter', 'gruppenleiter',
            'geschäftsführ', 'geschäftsleitung',
            'vorstand', 'aufsichtsrat', 'beirat',
            'direktor', 'director',
            'manager', 'leiter', 'leiterin',
            'chef', 'chefin',
            'head of', 'senior', 'junior',
            'assistent', 'assistentin', 'sekretär',
            'mitarbeiter', 'angestellte',
            'partner', 'gesellschafter',
            'inhaber', 'inhaberin', 'eigentümer',
            'gründer', 'gründerin', 'founder',
            'ceo', 'cto', 'cfo', 'coo', 'cmo', 'cio',
            'managing', 'executive', 'officer',
            'consultant', 'berater', 'beraterin',
            'entwickler', 'developer', 'engineer',
            'und team', 'unser team', 'das team',
        ]

        # Check if text is primarily a job title (not a real name)
        if any(p in text_lower for p in job_title_patterns):
            return False

        # First word should start with uppercase
        if not words[0][0].isupper():
            return False

        # Additional check: both words should look like names (capitalized, letters only)
        for word in words[:2]:  # Check at least first two words
            # Allow hyphens in names (e.g., "Hans-Peter")
            clean_word = word.replace('-', '')
            if not clean_word.isalpha():
                return False
            if not word[0].isupper():
                return False

        return True

    def _prioritize_team_members(
        self,
        members: List[TeamMember],
        job_category: Optional[str]
    ) -> List[TeamMember]:
        """Prioritize team members by relevance."""
        hr_keywords = ['personal', 'hr', 'recruiting', 'human', 'bewerbung']
        exec_keywords = ['geschäftsführ', 'ceo', 'inhaber', 'founder', 'gründer', 'managing']

        # Category-specific keywords
        category_keywords = {}
        if job_category:
            cat_lower = job_category.lower()
            if 'it' in cat_lower or 'tech' in cat_lower or 'software' in cat_lower:
                category_keywords = ['cto', 'it', 'tech', 'entwickl', 'engineer']
            elif 'sales' in cat_lower or 'vertrieb' in cat_lower:
                category_keywords = ['sales', 'vertrieb', 'verkauf']
            elif 'marketing' in cat_lower:
                category_keywords = ['marketing', 'cmo', 'kommunikation']

        def score(member: TeamMember) -> int:
            s = 0
            title_lower = (member.title or "").lower()

            # HR/Recruiting = highest priority
            if any(k in title_lower for k in hr_keywords):
                s += 100

            # Category-relevant
            if category_keywords and any(k in title_lower for k in category_keywords):
                s += 50

            # Executives = fallback
            if any(k in title_lower for k in exec_keywords):
                s += 25

            return s

        return sorted(members, key=score, reverse=True)

    async def _google_search(
        self,
        company_name: str,
        domain: Optional[str]
    ) -> Optional[str]:
        """Use Google Custom Search to find Impressum page."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            query = f'"{company_name}" impressum'
            if domain:
                query += f" site:{domain}"

            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_api_key,
                "cx": self.google_cse_id,
                "q": query,
                "num": 3
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                for item in items:
                    link = item.get("link", "")
                    # Prefer Impressum pages
                    if "impressum" in link.lower():
                        return link
                    # Or kontakt pages
                    if "kontakt" in link.lower():
                        return link

                # Return first result if no Impressum found
                if items:
                    return items[0].get("link")

            except Exception as e:
                logger.warning(f"Google search failed: {e}")

            return None

    async def _scrape_url(self, url: str) -> Optional[ImpressumResult]:
        """Scrape a URL for contact information."""
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        ) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")
                text = soup.get_text(separator=" ")

                phones = self._extract_phones(text)
                emails = self._extract_emails(text)
                address = self._extract_address(text)

                # Extract base website URL from the scraped page
                website_url = self._get_base_url(str(response.url))

                logger.info(f"Impressum {url}: {len(phones)} phones, {len(emails)} emails, address={address is not None}")

                return ImpressumResult(
                    phones=phones,
                    emails=emails,
                    website_url=website_url,
                    address=address,
                    raw_text=text[:10000],  # Keep raw text for AI extraction
                    success=len(phones) > 0 or len(emails) > 0 or address is not None
                )

            except httpx.HTTPStatusError as e:
                logger.debug(f"Impressum page not found: {url} ({e.response.status_code})")
                return None
            except Exception as e:
                logger.debug(f"Impressum scrape failed: {url} - {e}")
                return None

    def _extract_phones(self, text: str) -> List[PhoneResult]:
        """Extract phone numbers from text."""
        phones = []
        seen = set()

        # German phone patterns - order matters, more specific first
        patterns = [
            # +49 format with full number
            r'\+49\s*\(?\d{1,4}\)?\s*[\d\s\-/\.]{6,}',
            # 0049 format
            r'0049\s*\(?\d{1,4}\)?\s*[\d\s\-/\.]{6,}',
            # Local format with area code (0xxx followed by number)
            r'0[1-9]\d{2,4}\s*[-/\s\.]*\d{2,}[\d\s\-/\.]*',
            # Labeled patterns - capture the full number after label
            r'(?:Tel(?:efon)?|Phone|Fon|Mobil|Handy|Telefax)[:\.\s]+\+?[\d\s\-/\.\(\)]{8,}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the number
                number = self._clean_phone_number(match)

                # Validate phone number length
                # German numbers: +49 + max 12 digits = max 15 chars
                # Min: +49 + 8 digits = 11 chars
                digits_only = re.sub(r'\D', '', number)
                if not number or len(digits_only) < 10 or len(digits_only) > 15:
                    logger.debug(f"Invalid phone length: {number} ({len(digits_only)} digits)")
                    continue

                if number in seen:
                    continue

                seen.add(number)
                phone_type = self._determine_phone_type(number)

                # Context note based on phone type
                if phone_type == PhoneType.MOBILE:
                    context = "Auf Impressum gefunden - wahrscheinlich geschäftlich (Mobil)"
                elif phone_type == PhoneType.LANDLINE:
                    context = "Auf Impressum gefunden - geschäftliche Festnetznummer"
                else:
                    context = "Auf Impressum gefunden - wahrscheinlich geschäftlich"

                phones.append(PhoneResult(
                    number=number,
                    type=phone_type,
                    source=PhoneSource.IMPRESSUM,
                    context_note=context
                ))

        return phones

    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        pattern = r'[\w\.\-+]+@[\w\.\-]+\.[a-zA-Z]{2,}'
        emails = re.findall(pattern, text)

        # Filter out common false positives
        filtered = []
        for email in emails:
            email_lower = email.lower()
            # Skip image files and common non-contact emails
            if any(x in email_lower for x in ['.png', '.jpg', '.gif', 'example.com']):
                continue
            filtered.append(email)

        return list(set(filtered))

    def _clean_phone_number(self, raw: str) -> str:
        """Clean and normalize phone number."""
        # Remove label text
        raw = re.sub(r'^.*?(?=[\d+])', '', raw)
        # Keep only digits and +
        cleaned = re.sub(r'[^\d+]', '', raw)

        # Normalize to +49 format
        if cleaned.startswith('+49'):
            # Remove extra 0 after +49 if present
            cleaned = re.sub(r'^\+490', '+49', cleaned)
        elif cleaned.startswith('0049'):
            # Convert 0049 to +49
            cleaned = '+49' + cleaned[4:].lstrip('0')
        elif cleaned.startswith('0'):
            # Convert leading 0 to +49
            cleaned = '+49' + cleaned[1:]
        elif len(cleaned) >= 10 and not cleaned.startswith('+'):
            # Assume German number without prefix, add +49
            cleaned = '+49' + cleaned

        # FIX: Remove duplicate country code (+4949 -> +49)
        # This happens when source has formats like "+49 (0)49 89..." or "0049 49 89..."
        cleaned = re.sub(r'^\+4949', '+49', cleaned)

        return cleaned

    def _determine_phone_type(self, number: str) -> PhoneType:
        """Determine if phone is mobile or landline based on German prefixes."""
        # German mobile prefixes: 015x, 016x, 017x
        if re.match(r'(\+49|0049)?1[567]\d', number):
            return PhoneType.MOBILE

        # Austrian mobile: +43 6xx
        if re.match(r'(\+43|0043)?6\d', number):
            return PhoneType.MOBILE

        # Swiss mobile: +41 7x
        if re.match(r'(\+41|0041)?7[6789]\d', number):
            return PhoneType.MOBILE

        # Has country code or starts with 0 + area code = likely landline
        if re.match(r'(\+\d{2}|0\d{2,5})', number):
            return PhoneType.LANDLINE

        return PhoneType.UNKNOWN

    def _extract_address(self, text: str) -> Optional[str]:
        """Extract street address from Impressum text."""
        # German address patterns: Street + Number, PLZ + City
        # e.g. "Musterstraße 123, 12345 Berlin"
        patterns = [
            # Full address: Street Number, PLZ City
            r'([A-ZÄÖÜ][a-zäöüß]+(?:straße|str\.|weg|platz|allee|ring|gasse|damm)\s+\d+[a-z]?\s*,?\s*\d{5}\s+[A-ZÄÖÜ][a-zäöüß\-]+)',
            # Street + Number only
            r'([A-ZÄÖÜ][a-zäöüß]+(?:straße|str\.|weg|platz|allee|ring|gasse|damm)\s+\d+[a-z]?)',
            # PLZ + City pattern
            r'(\d{5}\s+[A-ZÄÖÜ][a-zäöüß\-]+(?:\s+[A-ZÄÖÜ][a-zäöüß\-]+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                # Validate: should have numbers and letters
                if re.search(r'\d', address) and re.search(r'[a-zA-Z]', address):
                    return address

        return None

    def _get_base_url(self, url: str) -> str:
        """Extract base website URL from a full URL."""
        # https://www.example.com/impressum -> https://www.example.com
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
