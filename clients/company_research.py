"""Company Research - Free company intelligence for sales calls."""
import logging
import re
import httpx
from typing import Optional, List
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

from config import get_settings
from utils.cost_tracker import track_llm

logger = logging.getLogger(__name__)


@dataclass
class CompanyIntel:
    """Company intelligence for sales preparation."""
    summary: str = ""  # AI-generated sales brief
    description: str = ""  # What the company does
    industry: str = ""
    employee_count: Optional[str] = None
    founded: Optional[str] = None
    headquarters: str = ""
    products_services: List[str] = field(default_factory=list)
    recent_news: List[str] = field(default_factory=list)
    hiring_signals: List[str] = field(default_factory=list)  # Growth indicators
    website_url: str = ""
    linkedin_url: str = ""
    raw_about_text: str = ""  # For debugging


class CompanyResearcher:
    """Research company information from free sources."""

    def __init__(self):
        settings = get_settings()
        self.timeout = settings.api_timeout
        self.anthropic_key = settings.anthropic_api_key

    async def research(
        self,
        company_name: str,
        domain: Optional[str] = None,
        job_description: Optional[str] = None,
        job_title: Optional[str] = None
    ) -> CompanyIntel:
        """
        Research company from multiple free sources.

        Args:
            company_name: Company name
            domain: Company domain (e.g., ten31.com)
            job_description: Job posting text (contains company info)
            job_title: Job title being hired for
        """
        intel = CompanyIntel()

        # Step 1: Scrape company website
        if domain:
            intel.website_url = f"https://{domain}"
            about_text = await self._scrape_about_page(domain)
            if about_text:
                intel.raw_about_text = about_text
                logger.info(f"Scraped about page: {len(about_text)} chars")

        # Step 2: Extract structured data from website
        if intel.raw_about_text:
            extracted = self._extract_company_data(intel.raw_about_text)
            intel.description = extracted.get("description", "")
            intel.founded = extracted.get("founded")
            intel.employee_count = extracted.get("employees")

        # Step 3: Analyze hiring signals from job posting
        if job_description and job_title:
            intel.hiring_signals = self._analyze_hiring_signals(job_description, job_title)

        # Step 4: Generate AI sales brief
        intel.summary = await self._generate_sales_brief(
            company_name=company_name,
            about_text=intel.raw_about_text,
            job_description=job_description,
            job_title=job_title,
            hiring_signals=intel.hiring_signals
        )

        return intel

    async def _scrape_about_page(self, domain: str) -> str:
        """Scrape company about/über uns page."""
        # Common about page paths for German companies
        about_paths = [
            "/ueber-uns",
            "/uber-uns",
            "/about",
            "/about-us",
            "/unternehmen",
            "/company",
            "/wir",
            "/team",
            "/",  # Homepage often has company description
        ]

        combined_text = []

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LeadEnrichBot/1.0)"}
        ) as client:
            for path in about_paths[:4]:  # Limit to first 4 paths
                url = f"https://{domain}{path}"
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        text = self._extract_text_from_html(response.text)
                        if text and len(text) > 100:
                            combined_text.append(text)
                            logger.debug(f"Scraped {url}: {len(text)} chars")
                except Exception as e:
                    logger.debug(f"Failed to scrape {url}: {e}")
                    continue

        return "\n\n---\n\n".join(combined_text)

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator=" ", strip=True)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)

        # Limit length
        return text[:8000] if text else ""

    def _extract_company_data(self, text: str) -> dict:
        """Extract structured data from about page text."""
        data = {}

        # Extract founding year
        year_patterns = [
            r'gegründet\s*(?:im\s*Jahr\s*)?(\d{4})',
            r'seit\s+(\d{4})',
            r'founded\s*(?:in\s*)?(\d{4})',
            r'established\s*(?:in\s*)?(\d{4})',
        ]
        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["founded"] = match.group(1)
                break

        # Extract employee count
        employee_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:Mitarbeiter|Mitarbeitende|Angestellte|employees)',
            r'(?:über|mehr als|around|over)\s*(\d+)\s*(?:Mitarbeiter|employees)',
            r'team\s*(?:von|of)\s*(\d+)',
        ]
        for pattern in employee_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data["employees"] = match.group(1)
                break

        # Get first 500 chars as description
        if text:
            data["description"] = text[:500].strip()

        return data

    def _analyze_hiring_signals(self, job_description: str, job_title: str) -> List[str]:
        """Analyze job posting for sales-relevant signals."""
        signals = []
        text_lower = job_description.lower()
        title_lower = job_title.lower()

        # Growth signals
        if any(word in text_lower for word in ["wachstum", "growth", "expanding", "wachsend"]):
            signals.append("Unternehmen im Wachstum")

        if any(word in text_lower for word in ["neu gegründet", "startup", "jung", "young company"]):
            signals.append("Junges/neues Unternehmen")

        # Hiring urgency
        if any(word in text_lower for word in ["sofort", "ab sofort", "immediately", "asap"]):
            signals.append("Dringende Einstellung")

        # Team expansion
        if any(word in text_lower for word in ["team verstärk", "team erweiter", "team aufbau"]):
            signals.append("Team wird ausgebaut")

        # Leadership hire
        if any(word in title_lower for word in ["head", "lead", "manager", "director", "leiter"]):
            signals.append("Führungsposition wird besetzt")

        # Senior hire
        if any(word in title_lower for word in ["senior", "experienced", "erfahren"]):
            signals.append("Erfahrene Position (Senior)")

        # Remote/modern workplace
        if any(word in text_lower for word in ["remote", "homeoffice", "home office", "hybrid"]):
            signals.append("Moderne Arbeitsplatzkultur (Remote/Hybrid)")

        # Good benefits mentioned
        benefits = []
        if "betriebliche altersvorsorge" in text_lower:
            benefits.append("bAV")
        if "30 tage urlaub" in text_lower or "30 urlaubstage" in text_lower:
            benefits.append("30 Tage Urlaub")
        if benefits:
            signals.append(f"Attraktive Benefits: {', '.join(benefits)}")

        return signals

    async def _generate_sales_brief(
        self,
        company_name: str,
        about_text: str,
        job_description: Optional[str],
        job_title: Optional[str],
        hiring_signals: List[str]
    ) -> str:
        """Generate AI sales brief using Claude."""
        if not self.anthropic_key:
            return self._generate_fallback_brief(company_name, about_text, hiring_signals)

        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.anthropic_key)

            prompt = f"""Du bist ein Sales Research Assistant für eine Personalberatung.
Erstelle eine kurze, prägnante Zusammenfassung für einen Sales Call.

UNTERNEHMEN: {company_name}

ÜBER DAS UNTERNEHMEN (von Website):
{about_text[:3000] if about_text else "Keine Informationen verfügbar"}

AKTUELLE STELLENANZEIGE:
Titel: {job_title or "N/A"}
{job_description[:2000] if job_description else ""}

ERKANNTE HIRING-SIGNALE:
{chr(10).join(f"- {s}" for s in hiring_signals) if hiring_signals else "Keine besonderen Signale"}

---

Erstelle eine Sales-Zusammenfassung mit:
1. **Was macht das Unternehmen?** (1-2 Sätze)
2. **Branche & Größe** (wenn erkennbar)
3. **Aktuelle Situation** (warum stellen sie ein?)
4. **Gesprächseinstieg** (ein konkreter Aufhänger für den Call)

Halte es kurz und actionable (max 150 Wörter). Schreibe auf Deutsch."""

            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            track_llm("company_research", tier="sonnet")  # Track company research LLM call

            return response.content[0].text.strip()

        except Exception as e:
            logger.error(f"Failed to generate AI brief: {e}")
            return self._generate_fallback_brief(company_name, about_text, hiring_signals)

    def _generate_fallback_brief(
        self,
        company_name: str,
        about_text: str,
        hiring_signals: List[str]
    ) -> str:
        """Generate simple brief without AI."""
        parts = [f"**{company_name}**"]

        if about_text:
            # First sentence
            first_sentence = about_text.split('.')[0][:200]
            parts.append(f"\n{first_sentence}.")

        if hiring_signals:
            parts.append(f"\n\nSignale: {', '.join(hiring_signals)}")

        return "".join(parts)


async def research_company(
    company_name: str,
    domain: Optional[str] = None,
    job_description: Optional[str] = None,
    job_title: Optional[str] = None
) -> CompanyIntel:
    """Convenience function for company research."""
    researcher = CompanyResearcher()
    return await researcher.research(
        company_name=company_name,
        domain=domain,
        job_description=job_description,
        job_title=job_title
    )
