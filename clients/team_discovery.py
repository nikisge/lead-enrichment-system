"""
Team Page Discovery for Lead Enrichment.

Uses Google Search + AI to find team/management pages:
1. Google search for team-related pages
2. AI analyzes snippets to find best URLs
3. Scrapes promising pages
4. AI extracts contacts

Fallback: LinkedIn search for decision makers if no team page found.
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from config import get_settings
from clients.llm_client import get_llm_client, ModelTier
from clients.ai_extractor import extract_contacts_from_page, ExtractedContact
from clients.ai_validator import validate_linkedin_match
from utils.cost_tracker import track_google, track_llm

logger = logging.getLogger(__name__)

# Maximum page size to scrape (50KB)
MAX_PAGE_SIZE = 50_000

# Maximum text to extract from page
MAX_TEXT_EXTRACT = 30_000


@dataclass
class DiscoveredPage:
    """A page discovered via Google search."""
    url: str
    title: str
    snippet: str
    relevance_score: float = 0.0


@dataclass
class TeamDiscoveryResult:
    """Result from team discovery process."""
    contacts: List[ExtractedContact]
    source_urls: List[str]
    fallback_used: bool = False
    success: bool = False


class TeamDiscovery:
    """
    Discovers team/management pages and extracts contacts.
    """

    def __init__(self):
        settings = get_settings()
        self.google_api_key = settings.google_api_key
        self.google_cse_id = settings.google_cse_id
        self.timeout = settings.api_timeout

    async def discover_and_extract(
        self,
        company_name: str,
        domain: Optional[str] = None,
        job_category: Optional[str] = None,
        max_pages: int = 2
    ) -> TeamDiscoveryResult:
        """
        Full discovery process: Find pages, scrape, extract contacts.

        Args:
            company_name: Company name
            domain: Company domain
            job_category: Job category for relevance
            max_pages: Maximum pages to scrape

        Returns:
            TeamDiscoveryResult with contacts and metadata
        """
        logger.info(f"Starting team discovery for {company_name}")

        # Step 1: Find team page URLs via Google
        discovered_pages = await self._discover_team_pages(company_name, domain)

        if not discovered_pages:
            logger.info(f"No team pages found for {company_name}, trying fallback")
            # Fallback: LinkedIn search for decision makers
            contacts = await self._fallback_linkedin_search(company_name, job_category)
            return TeamDiscoveryResult(
                contacts=contacts,
                source_urls=[],
                fallback_used=True,
                success=len(contacts) > 0
            )

        # Step 2: Scrape top pages
        all_contacts = []
        scraped_urls = []

        for page in discovered_pages[:max_pages]:
            logger.info(f"Scraping team page: {page.url}")
            contacts = await self._scrape_and_extract(page.url, company_name)

            if contacts:
                all_contacts.extend(contacts)
                scraped_urls.append(page.url)
                logger.info(f"Found {len(contacts)} contacts from {page.url}")

        # Deduplicate by name
        unique_contacts = self._deduplicate_contacts(all_contacts)

        # If still no contacts, try fallback
        if not unique_contacts:
            logger.info("No contacts extracted, trying LinkedIn fallback")
            unique_contacts = await self._fallback_linkedin_search(company_name, job_category)
            return TeamDiscoveryResult(
                contacts=unique_contacts,
                source_urls=scraped_urls,
                fallback_used=True,
                success=len(unique_contacts) > 0
            )

        return TeamDiscoveryResult(
            contacts=unique_contacts,
            source_urls=scraped_urls,
            fallback_used=False,
            success=True
        )

    async def _discover_team_pages(
        self,
        company_name: str,
        domain: Optional[str] = None
    ) -> List[DiscoveredPage]:
        """
        Find team/management pages via Google Search.

        Searches:
        1. "{company}" Team Geschäftsführung site:{domain}
        2. "{company}" Ansprechpartner Mitarbeiter
        3. "{company}" über uns Team
        """
        if not self.google_api_key or not self.google_cse_id:
            logger.warning("Google API not configured for team discovery")
            return []

        # Build search queries
        queries = []

        if domain:
            # Site-specific search first
            queries.append(f'"{company_name}" Team OR Geschäftsführung OR Ansprechpartner site:{domain}')

        # General searches
        queries.extend([
            f'"{company_name}" Team Geschäftsführung',
            f'"{company_name}" Ansprechpartner über uns',
            f'"{company_name}" Mitarbeiter Kontakt'
        ])

        all_results = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for query in queries[:3]:  # Max 3 queries
                results = await self._google_search(client, query)
                all_results.extend(results)

                # Stop if we have enough results
                if len(all_results) >= 10:
                    break

        if not all_results:
            return []

        # Use AI to rank and filter results
        return await self._analyze_search_results(all_results, company_name, domain)

    async def _google_search(
        self,
        client: httpx.AsyncClient,
        query: str,
        num_results: int = 5
    ) -> List[Dict[str, str]]:
        """Execute Google Custom Search."""
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": num_results
        }

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            track_google("team_discovery_search")  # Track Google API call
            data = response.json()

            results = []
            for item in data.get("items", []):
                results.append({
                    "url": item.get("link", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", "")
                })

            return results

        except Exception as e:
            logger.warning(f"Google search failed: {e}")
            return []

    async def _analyze_search_results(
        self,
        results: List[Dict[str, str]],
        company_name: str,
        domain: Optional[str]
    ) -> List[DiscoveredPage]:
        """
        Use AI to analyze search results and find best team page URLs.
        """
        if not results:
            return []

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Filter out obvious non-team pages
        filtered = []
        skip_domains = [
            'linkedin.com', 'xing.com', 'facebook.com', 'twitter.com',
            'youtube.com', 'instagram.com', 'kununu.de', 'glassdoor.com',
            'indeed.com', 'stepstone.de', 'monster.de', 'arbeitsagentur.de'
        ]

        for r in unique_results:
            url = r.get("url", "").lower()
            if not any(skip in url for skip in skip_domains):
                filtered.append(r)

        if not filtered:
            return []

        llm = get_llm_client()

        prompt = f"""Analysiere diese Google-Suchergebnisse für "{company_name}".
Welche URLs führen am wahrscheinlichsten zu einer Seite mit Team-Mitgliedern oder Ansprechpartnern?

Suchergebnisse:
{filtered[:10]}

Bewerte jede URL:
- Team/Mitarbeiter/Über-uns Seiten: hoch (0.8-1.0)
- Kontakt-Seiten: mittel (0.5-0.7)
- Impressum: niedrig (0.3-0.5)
- Irrelevante Seiten: sehr niedrig (0.0-0.2)

{f"Bevorzuge URLs von der Domain: {domain}" if domain else ""}

Antworte als JSON-Array, sortiert nach Relevanz:
[{{"url": "...", "title": "...", "snippet": "...", "relevance_score": 0.9}}]

Nur die top 3-4 relevantesten zurückgeben."""

        result = await llm.call_json(prompt, tier=ModelTier.FAST)
        track_llm("team_rank", tier="haiku")  # Ranking uses Haiku

        if not result or not isinstance(result, list):
            # Fallback: return filtered results without AI ranking
            return [
                DiscoveredPage(
                    url=r["url"],
                    title=r["title"],
                    snippet=r["snippet"],
                    relevance_score=0.5
                )
                for r in filtered[:3]
            ]

        # Parse AI results
        pages = []
        for item in result:
            if isinstance(item, dict) and item.get("url"):
                score = item.get("relevance_score", 0.5)
                if score >= 0.3:  # Minimum threshold
                    pages.append(DiscoveredPage(
                        url=item["url"],
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        relevance_score=score
                    ))

        # Sort by relevance
        pages.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"Found {len(pages)} relevant team pages")
        return pages

    async def _scrape_and_extract(
        self,
        url: str,
        company_name: str
    ) -> List[ExtractedContact]:
        """
        Scrape a URL and extract contacts using AI.
        Uses Playwright for JS-rendering (most team pages are JS-heavy).
        """
        html = await self._scrape_with_playwright(url)
        source = "playwright"

        if not html:
            # Fallback to httpx for simple sites
            html = await self._scrape_with_httpx(url)
            source = "httpx"

        if not html:
            logger.warning(f"Failed to scrape {url}")
            return []

        logger.debug(f"Got {len(html)} bytes HTML from {url} via {source}")

        # Parse and extract text
        soup = BeautifulSoup(html, "lxml")

        # Log what elements we found before removing
        body = soup.find('body')
        if body:
            all_text_before = body.get_text(separator=" ", strip=True)
            logger.debug(f"Body text before cleanup: {len(all_text_before)} chars")

        # Remove non-content elements
        for elem in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
            elem.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # If text is very short, the page is likely a JS SPA that didn't render
        if len(text) < 100:
            logger.warning(f"Very little text extracted ({len(text)} chars) from {url} - likely JS SPA not fully rendered")

            # Try to find any useful content in specific containers
            for selector in ['main', 'article', '.content', '#content', '#app', '#root', '.page']:
                container = soup.select_one(selector)
                if container:
                    container_text = container.get_text(separator="\n", strip=True)
                    if len(container_text) > len(text):
                        text = container_text
                        logger.debug(f"Found more text in {selector}: {len(text)} chars")

        # Truncate if needed
        if len(text) > MAX_TEXT_EXTRACT:
            text = text[:MAX_TEXT_EXTRACT]

        logger.info(f"Extracted {len(text)} chars from {url}")

        # Skip AI extraction if we have almost no text
        if len(text) < 50:
            logger.warning(f"Skipping AI extraction - not enough text ({len(text)} chars)")
            return []

        # Use AI to extract contacts
        return await extract_contacts_from_page(text, company_name, "team")

    async def _scrape_with_playwright(self, url: str) -> Optional[str]:
        """Scrape URL with Playwright for JS-rendering."""
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        viewport={'width': 1280, 'height': 720},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    page = await context.new_page()

                    # Navigate and wait for network to be idle (JS finished loading)
                    try:
                        await page.goto(url, wait_until='networkidle', timeout=self.timeout * 1000)
                    except Exception:
                        # Fallback to domcontentloaded if networkidle times out
                        logger.debug(f"networkidle timeout for {url}, trying domcontentloaded")
                        await page.goto(url, wait_until='domcontentloaded', timeout=self.timeout * 1000)

                    # Wait for content to appear - try common selectors
                    content_selectors = ['main', 'article', '.content', '#content', '.main-content', '[role="main"]']
                    content_found = False

                    for selector in content_selectors:
                        try:
                            await page.wait_for_selector(selector, timeout=3000)
                            content_found = True
                            logger.debug(f"Found content selector: {selector}")
                            break
                        except Exception:
                            continue

                    # If no content selector found, wait a bit longer for JS
                    if not content_found:
                        logger.debug(f"No content selector found, waiting 4s for JS")
                        await page.wait_for_timeout(4000)
                    else:
                        # Small additional wait for any lazy-loaded content
                        await page.wait_for_timeout(1000)

                    # Scroll down to trigger lazy loading
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                    await page.wait_for_timeout(500)

                    html = await page.content()
                    logger.debug(f"Playwright got {len(html)} bytes from {url}")

                    if len(html) > MAX_PAGE_SIZE:
                        html = html[:MAX_PAGE_SIZE]

                    return html

                finally:
                    await browser.close()

        except ImportError:
            logger.warning("Playwright not installed, falling back to httpx")
            return None
        except Exception as e:
            logger.warning(f"Playwright scraping failed for {url}: {e}")
            return None

    async def _scrape_with_httpx(self, url: str) -> Optional[str]:
        """Fallback scraping with httpx (no JS rendering)."""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > MAX_PAGE_SIZE * 2:
                        logger.warning(f"Page too large: {content_length} bytes")
                        return None

                    chunks = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_PAGE_SIZE:
                            break
                        chunks.append(chunk)

                    return b"".join(chunks).decode("utf-8", errors="ignore")

        except Exception as e:
            logger.warning(f"httpx scraping failed for {url}: {e}")
            return None

    async def _fallback_linkedin_search(
        self,
        company_name: str,
        job_category: Optional[str] = None
    ) -> List[ExtractedContact]:
        """
        Fallback: Search LinkedIn for decision makers when no team page found.

        Searches for:
        - Geschäftsführer
        - HR Manager / Personalleiter
        - Department heads based on job category
        """
        if not self.google_api_key or not self.google_cse_id:
            return []

        # Determine positions to search
        positions = ["Geschäftsführer", "HR Manager", "Personalleiter"]

        if job_category:
            cat_lower = job_category.lower()
            if "it" in cat_lower or "software" in cat_lower or "tech" in cat_lower:
                positions.insert(1, "CTO")
            elif "sales" in cat_lower or "vertrieb" in cat_lower:
                positions.insert(1, "Vertriebsleiter")
            elif "marketing" in cat_lower:
                positions.insert(1, "Marketing-Leiter")

        contacts = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for position in positions[:3]:  # Max 3 position searches
                query = f'"{company_name}" "{position}" site:linkedin.com/in'
                results = await self._google_search(client, query, num_results=3)

                for result in results:
                    url = result.get("url", "")
                    if "linkedin.com/in/" not in url:
                        continue

                    # Validate with AI
                    validation = await validate_linkedin_match(
                        linkedin_snippet=result.get("snippet", ""),
                        linkedin_title=result.get("title", ""),
                        person_name="",  # We don't know the name yet
                        company_name=company_name
                    )

                    if validation.valid:
                        # Extract name from LinkedIn title
                        name = self._extract_name_from_linkedin_title(result.get("title", ""))
                        if name:
                            contacts.append(ExtractedContact(
                                name=name,
                                title=position,
                                source="linkedin_fallback"
                            ))

                # Stop if we have enough
                if len(contacts) >= 3:
                    break

        logger.info(f"LinkedIn fallback found {len(contacts)} contacts")
        return contacts

    def _extract_name_from_linkedin_title(self, title: str) -> Optional[str]:
        """Extract name from LinkedIn title like 'Max Müller - HR Manager | LinkedIn'."""
        if not title:
            return None

        import re

        # Remove " | LinkedIn" suffix
        title = re.sub(r'\s*\|\s*LinkedIn.*$', '', title, flags=re.IGNORECASE)

        # Split by " - " to separate name from title
        parts = re.split(r'\s*[-–]\s*', title)
        if parts:
            name = parts[0].strip()
            # Validate: at least 2 words
            if len(name.split()) >= 2:
                return name

        return None

    def _deduplicate_contacts(self, contacts: List[ExtractedContact]) -> List[ExtractedContact]:
        """Remove duplicate contacts by name."""
        seen_names = set()
        unique = []

        for contact in contacts:
            name_lower = contact.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique.append(contact)

        return unique


# Convenience function
async def discover_team_contacts(
    company_name: str,
    domain: Optional[str] = None,
    job_category: Optional[str] = None
) -> TeamDiscoveryResult:
    """
    Discover team contacts for a company.

    Convenience function for quick access.
    """
    discovery = TeamDiscovery()
    return await discovery.discover_and_extract(
        company_name=company_name,
        domain=domain,
        job_category=job_category
    )
