"""
Team Page Discovery for Lead Enrichment - V2.

Smart "2-Klicks" approach to find team pages like a human would:

1. Direct URL Check - Try common team page URLs (/team, /ueber-uns, etc.)
2. Sitemap Scan - Parse sitemap.xml for team-related URLs
3. Homepage Link Scan - Find team/about links on homepage (no AI, just pattern matching)
4. Improved Scraping - Better Playwright waits for JS-heavy team pages

Goal: Find the same contacts a human would find with "2 clicks" on the company website.

Cost: $0 extra (no AI for URL discovery, only for contact extraction)
Time: ~25-30 seconds
"""

import logging
import asyncio
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET

import httpx
from bs4 import BeautifulSoup

from config import get_settings
from clients.ai_extractor import extract_contacts_from_page, ExtractedContact

logger = logging.getLogger(__name__)

# Maximum page size to scrape (100KB for team pages - they can be image-heavy)
MAX_PAGE_SIZE = 100_000

# Maximum text to extract from page
MAX_TEXT_EXTRACT = 30_000

# Common team page URL patterns (ordered by priority)
TEAM_URL_PATTERNS = [
    "/team",
    "/unser-team",
    "/das-team",
    "/ueber-uns",
    "/uber-uns",
    "/about-us",
    "/about",
    "/ueber-uns/team",
    "/about/team",
    "/unternehmen/team",
    "/ansprechpartner",
    "/kontakt",
    "/contact",
    "/mitarbeiter",
    "/menschen",
    "/people",
    "/wir",
    "/management",
    "/geschaeftsfuehrung",
    "/geschaeftsleitung",
    "/fuehrungsteam",
    "/leadership",
    "/unternehmen",
    "/firma",
    "/company",
    "/who-we-are",
    "/wer-wir-sind",
]

# Keywords to find team links on homepage (German + English)
TEAM_LINK_KEYWORDS = [
    "team", "über uns", "ueber uns", "about us", "about",
    "ansprechpartner", "kontakt", "contact", "mitarbeiter",
    "menschen", "people", "wir über uns", "das sind wir",
    "management", "geschäftsführung", "leadership", "unternehmen"
]

# Team-specific CSS selectors to wait for
TEAM_PAGE_SELECTORS = [
    ".team-member", ".team-card", ".employee", ".mitarbeiter",
    ".person", ".staff", ".member", ".ansprechpartner",
    "[class*='team']", "[class*='employee']", "[class*='member']",
    "[class*='person']", "[class*='staff']", "[class*='mitarbeiter']",
    ".leadership", ".management", ".geschaeftsfuehrung",
    # Common grid/card patterns
    ".person-card", ".team-grid", ".people-grid",
    ".card", ".profile-card", ".bio",
    "[class*='about']", "[class*='profile']",
    "[class*='card']", "[class*='grid'] [class*='item']",
    ".swiper-slide",  # Slider-based team pages
    "[data-team]", "[data-member]",  # Data attributes
    # WordPress patterns
    ".wp-block-team", ".elementor-team-member",
]


@dataclass
class DiscoveredPage:
    """A page discovered for team contact extraction."""
    url: str
    source: str  # "direct_url", "sitemap", "homepage_link"
    relevance_score: float = 0.0
    title: str = ""


@dataclass
class TeamDiscoveryResult:
    """Result from team discovery process."""
    contacts: List[ExtractedContact]
    source_urls: List[str]
    discovery_method: str = ""  # How we found the team page
    fallback_used: bool = False
    success: bool = False


class TeamDiscovery:
    """
    Smart team page discovery - finds contacts like a human with "2 clicks".
    """

    def __init__(self):
        settings = get_settings()
        self.timeout = settings.api_timeout
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=10,  # Quick timeout for URL checks
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
                }
            )
        return self._http_client

    async def discover_and_extract(
        self,
        company_name: str,
        domain: Optional[str] = None,
        job_category: Optional[str] = None,
        max_pages: int = 3,
        target_titles: Optional[List[str]] = None
    ) -> TeamDiscoveryResult:
        """
        Full discovery process: Find team pages smartly, scrape, extract contacts.

        Strategy (in order):
        1. Direct URL check - Try common /team, /ueber-uns URLs
        2. Sitemap scan - Parse sitemap.xml for team URLs
        3. Homepage link scan - Find team links on homepage
        4. Scrape found pages with improved Playwright

        Args:
            company_name: Company name
            domain: Company domain (required for website scraping)
            job_category: Job category for relevance

        Returns:
            TeamDiscoveryResult with contacts and metadata
        """
        logger.info(f"━━━ TEAM DISCOVERY START: {company_name} ━━━")

        if not domain:
            logger.warning(f"⚠️ No domain provided for {company_name} - cannot discover team pages")
            return TeamDiscoveryResult(
                contacts=[],
                source_urls=[],
                discovery_method="no_domain",
                success=False
            )

        base_url = f"https://{domain}"
        logger.info(f"🌐 Base URL: {base_url}")

        # Step 1: ALL discovery methods in parallel
        logger.info("📍 Step 1: Running all discovery methods in parallel...")
        direct_task = self._check_direct_urls(base_url)
        sitemap_task = self._scan_sitemap(base_url)
        homepage_task = self._scan_homepage_links(base_url)

        direct_pages_result, sitemap_pages_result, homepage_pages_result = await asyncio.gather(
            direct_task, sitemap_task, homepage_task, return_exceptions=True
        )

        discovered_pages = []

        if isinstance(direct_pages_result, list) and direct_pages_result:
            discovered_pages.extend(direct_pages_result)
            logger.info(f"✓ Found {len(direct_pages_result)} direct team page(s)")
            for page in direct_pages_result:
                logger.info(f"  → {page.url} (score: {page.relevance_score:.2f})")
        else:
            logger.info("✗ No direct team URLs found")

        if isinstance(sitemap_pages_result, list) and sitemap_pages_result:
            discovered_pages.extend(sitemap_pages_result)
            logger.info(f"✓ Found {len(sitemap_pages_result)} team page(s) in sitemap")
            for page in sitemap_pages_result:
                logger.info(f"  → {page.url}")
        else:
            logger.info("✗ No team pages in sitemap (or no sitemap)")

        if isinstance(homepage_pages_result, list) and homepage_pages_result:
            discovered_pages.extend(homepage_pages_result)
            logger.info(f"✓ Found {len(homepage_pages_result)} team link(s) on homepage")
            for page in homepage_pages_result:
                logger.info(f"  → {page.url} ('{page.title}')")
        else:
            logger.info("✗ No team links found on homepage")

        # Deduplicate and sort by relevance
        discovered_pages = self._deduplicate_pages(discovered_pages)
        discovered_pages.sort(key=lambda x: x.relevance_score, reverse=True)

        if not discovered_pages:
            logger.warning(f"❌ No team pages found for {company_name}")
            return TeamDiscoveryResult(
                contacts=[],
                source_urls=[],
                discovery_method="none_found",
                success=False
            )

        pages_to_scrape = discovered_pages[:max_pages]
        logger.info(f"📍 Step 2: Scraping {len(pages_to_scrape)} best page(s) in parallel...")
        for page in pages_to_scrape:
            logger.info(f"  → {page.url} (source: {page.source}, score: {page.relevance_score:.2f})")

        # Step 2: Scrape pages in parallel
        scrape_tasks = [
            self._scrape_team_page(
                page.url, company_name,
                target_titles=target_titles,
                job_category=job_category,
                company_domain=domain
            )
            for page in pages_to_scrape
        ]
        scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

        all_contacts = []
        scraped_urls = []
        discovery_methods = set()

        for page, result in zip(pages_to_scrape, scrape_results):
            if isinstance(result, Exception):
                logger.warning(f"  ✗ Scraping failed for {page.url}: {result}")
                continue
            if result:
                all_contacts.extend(result)
                scraped_urls.append(page.url)
                discovery_methods.add(page.source)
                logger.info(f"  ✓ {page.url}: {len(result)} contact(s)")
                for c in result:
                    logger.info(f"    → {c.name} ({c.title or 'no title'})")
            else:
                logger.info(f"  ✗ {page.url}: No contacts extracted")

        # Deduplicate contacts by name
        unique_contacts = self._deduplicate_contacts(all_contacts)

        method_str = "+".join(sorted(discovery_methods)) if discovery_methods else "none"

        logger.info(f"━━━ TEAM DISCOVERY COMPLETE: {len(unique_contacts)} contacts ━━━")

        return TeamDiscoveryResult(
            contacts=unique_contacts,
            source_urls=scraped_urls,
            discovery_method=method_str,
            fallback_used=False,
            success=len(unique_contacts) > 0
        )

    async def _check_direct_urls(self, base_url: str) -> List[DiscoveredPage]:
        """
        Check common team page URL patterns with HEAD requests.
        Fast and free - just checking if URLs exist.
        """
        client = await self._get_client()
        found_pages = []

        # Check URLs in parallel (fast)
        async def check_url(pattern: str, priority: int) -> Optional[DiscoveredPage]:
            url = f"{base_url}{pattern}"
            try:
                response = await client.head(url, timeout=5)
                if response.status_code == 200:
                    # Calculate relevance score based on pattern priority
                    score = 1.0 - (priority * 0.05)  # Earlier patterns = higher score
                    return DiscoveredPage(
                        url=url,
                        source="direct_url",
                        relevance_score=score
                    )
            except Exception as e:
                logger.debug(f"  HEAD {pattern}: failed ({e})")
            return None

        # Check all patterns in parallel
        tasks = [check_url(pattern, i) for i, pattern in enumerate(TEAM_URL_PATTERNS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, DiscoveredPage):
                found_pages.append(result)

        return found_pages

    async def _scan_sitemap(self, base_url: str) -> List[DiscoveredPage]:
        """
        Parse sitemap.xml to find team-related URLs.
        No AI needed - just XML parsing + keyword matching.
        """
        client = await self._get_client()
        sitemap_urls = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/sitemap-index.xml",
        ]

        for sitemap_url in sitemap_urls:
            try:
                response = await client.get(sitemap_url, timeout=10)
                if response.status_code != 200:
                    continue

                # Parse XML
                try:
                    root = ET.fromstring(response.text)
                except ET.ParseError:
                    logger.debug(f"Failed to parse sitemap: {sitemap_url}")
                    continue

                # Handle sitemap index (contains other sitemaps)
                namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

                # Check if this is a sitemap index
                sitemap_refs = root.findall('.//ns:sitemap/ns:loc', namespaces)
                if sitemap_refs:
                    # It's an index - check all sub-sitemaps (max 5) and accumulate results
                    all_sub_pages = []
                    for sitemap_ref in sitemap_refs[:5]:
                        sub_url = sitemap_ref.text
                        if sub_url:
                            sub_pages = await self._parse_sitemap_urls(sub_url, client)
                            all_sub_pages.extend(sub_pages)
                    if all_sub_pages:
                        return all_sub_pages

                # Parse URLs directly
                return await self._parse_sitemap_urls(sitemap_url, client, xml_content=response.text)

            except Exception as e:
                logger.debug(f"Sitemap check failed for {sitemap_url}: {e}")

        return []

    async def _parse_sitemap_urls(
        self,
        sitemap_url: str,
        client: httpx.AsyncClient,
        xml_content: Optional[str] = None
    ) -> List[DiscoveredPage]:
        """Parse a sitemap and find team-related URLs."""
        try:
            if xml_content is None:
                response = await client.get(sitemap_url, timeout=10)
                if response.status_code != 200:
                    return []
                xml_content = response.text

            root = ET.fromstring(xml_content)
            namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            pages = []
            urls = root.findall('.//ns:url/ns:loc', namespaces)

            # Also try without namespace (some sitemaps don't use it)
            if not urls:
                urls = root.findall('.//url/loc')

            # Skip patterns that indicate non-team pages
            SKIP_URL_PATTERNS = [
                '/blog/', '/news/', '/category/', '/tag/', '/author/',
                '/produkt', '/product', '/shop/', '/karriere/', '/career/',
                '/datenschutz', '/privacy', '/agb/', '/terms/',
                '/wp-content/', '/wp-json/', '/feed/',
            ]

            for url_elem in urls:
                url = url_elem.text
                if not url:
                    continue

                url_lower = url.lower()

                # Skip obvious non-team URLs
                if any(skip in url_lower for skip in SKIP_URL_PATTERNS):
                    continue

                # Check if URL contains team-related keywords
                team_keywords = [
                    '/team', '/unser-team', '/das-team',
                    '/ueber-uns', '/uber-uns', '/about-us', '/about/',
                    '/ansprechpartner', '/people', '/staff',
                    '/management', '/leadership', '/fuehrung',
                    '/geschaeftsfuehrung', '/vorstand', '/board',
                    '/who-we-are', '/wer-wir-sind',
                    '/mitarbeiter', '/menschen',
                    # Broader patterns (lower score)
                    '/ueber-', '/uber-', '/about',
                    '/unternehmen', '/company', '/firma',
                    '/kontakt', '/contact',
                ]

                for keyword in team_keywords:
                    if keyword in url_lower:
                        # Score based on keyword position in list (earlier = more relevant)
                        score = 0.9 - (team_keywords.index(keyword) * 0.03)
                        pages.append(DiscoveredPage(
                            url=url,
                            source="sitemap",
                            relevance_score=max(score, 0.1)
                        ))
                        break

            # If keyword matching found results, return them (sorted by score)
            if pages:
                pages.sort(key=lambda x: x.relevance_score, reverse=True)
                return pages[:5]

            # If keyword matching found 0 results but sitemap has many URLs,
            # use AI to select the best team/about pages
            all_urls = [
                url_elem.text for url_elem in urls
                if url_elem.text and not any(skip in url_elem.text.lower() for skip in SKIP_URL_PATTERNS)
            ]
            if len(all_urls) > 10:
                logger.info(f"  → No keyword matches in {len(all_urls)} sitemap URLs — trying AI selection")
                ai_pages = await self._ai_select_best_urls(all_urls)
                if ai_pages:
                    return ai_pages

            return pages[:5]  # Fallback (empty)

        except Exception as e:
            logger.debug(f"Sitemap parsing failed: {e}")
            return []

    async def _ai_select_best_urls(
        self,
        urls: List[str],
    ) -> List[DiscoveredPage]:
        """
        Use AI (FAST tier) to select the best team/about URLs from a list.
        Only called when keyword matching fails or is ambiguous.
        Sends only URL strings (tiny input, very cheap).
        """
        try:
            from clients.llm_client import get_llm_client, ModelTier

            llm = get_llm_client()

            # Only send path portions to save tokens
            urls_formatted = "\n".join(urls[:50])  # Cap at 50 URLs

            prompt = f"""Wähle aus dieser URL-Liste die 3 URLs, die am wahrscheinlichsten Seiten sind auf denen man Team-Mitglieder, Führungskräfte oder Ansprechpartner findet.

GUTE Seiten: /team, /ueber-uns, /about, /management, /leadership, /ansprechpartner, /people, /unternehmen
SCHLECHTE Seiten: /blog/..., /news/..., /produkte/..., /jobs/..., /karriere/..., /datenschutz, /agb

URLs:
{urls_formatted}

Antworte NUR als JSON-Array mit den 3 besten URLs (in Prioritäts-Reihenfolge):
["https://...", "https://...", "https://..."]

Falls keine passende URL dabei ist: []"""

            result = await llm.call_json(prompt, tier=ModelTier.FAST)
            try:
                from utils.cost_tracker import track_llm
                track_llm("sitemap_selection", tier="flash")
            except Exception:
                pass

            if not result or not isinstance(result, list):
                return []

            pages = []
            for i, url in enumerate(result[:3]):
                if isinstance(url, str) and url.startswith("http"):
                    pages.append(DiscoveredPage(
                        url=url,
                        source="sitemap",
                        relevance_score=0.85 - (i * 0.05)
                    ))

            if pages:
                logger.info(f"  ✓ AI selected {len(pages)} URL(s) from sitemap")
                for p in pages:
                    logger.info(f"    → {p.url}")

            return pages

        except Exception as e:
            logger.debug(f"AI sitemap selection failed: {e}")
            return []

    async def _scan_homepage_links(self, base_url: str) -> List[DiscoveredPage]:
        """
        Scan homepage for team/about links.
        No AI needed - just find <a> tags with team-related text.
        """
        client = await self._get_client()

        try:
            response = await client.get(base_url, timeout=15)
            if response.status_code != 200:
                logger.debug(f"Homepage request failed: {response.status_code}")
                return []

            soup = BeautifulSoup(response.text, 'lxml')

            # Find all links
            found_pages = []
            seen_urls = set()

            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True).lower()

                # Skip empty or javascript links
                if not href or href.startswith('#') or href.startswith('javascript:'):
                    continue

                # Make URL absolute
                full_url = urljoin(base_url, href)

                # Skip external links
                if not full_url.startswith(base_url):
                    continue

                # Skip if already seen
                if full_url in seen_urls:
                    continue

                # Check if link text or URL contains team keywords
                url_lower = full_url.lower()

                for keyword in TEAM_LINK_KEYWORDS:
                    if keyword in text or keyword.replace(' ', '-') in url_lower or keyword.replace(' ', '') in url_lower:
                        seen_urls.add(full_url)

                        # Calculate score based on keyword match
                        score = 0.7 - (TEAM_LINK_KEYWORDS.index(keyword) * 0.03)

                        # Boost if keyword is in link text (more reliable)
                        if keyword in text:
                            score += 0.1

                        found_pages.append(DiscoveredPage(
                            url=full_url,
                            source="homepage_link",
                            relevance_score=score,
                            title=link.get_text(strip=True)[:50]
                        ))
                        break

            # Sort by score and return top 5
            found_pages.sort(key=lambda x: x.relevance_score, reverse=True)
            return found_pages[:5]

        except Exception as e:
            logger.debug(f"Homepage scan failed: {e}")
            return []

    async def _scrape_team_page(
        self,
        url: str,
        company_name: str,
        target_titles: Optional[List[str]] = None,
        job_category: Optional[str] = None,
        company_domain: Optional[str] = None
    ) -> List[ExtractedContact]:
        """
        Scrape a team page with improved Playwright settings.

        Better than before:
        - Longer wait times for JS SPAs
        - Team-specific selector waiting
        - Full page scroll to trigger lazy loading
        - Structured text extraction preserving card layouts
        """
        result = await self._scrape_with_playwright_v2(url)
        html = None
        visible_text = ""

        if isinstance(result, tuple):
            html, visible_text = result
        elif isinstance(result, str):
            html = result

        if not html and not visible_text:
            logger.warning(f"⚠️ Playwright scrape failed, trying httpx fallback")
            html = await self._scrape_with_httpx(url)

        if not html and not visible_text:
            logger.warning(f"❌ All scraping methods failed for {url}")
            return []

        # Parse HTML for card detection
        card_header = ""
        if html:
            soup = BeautifulSoup(html, "lxml")
            for elem in soup(["script", "style", "noscript", "svg", "iframe"]):
                elem.decompose()
            card_header = self._try_extract_cards(soup)

        # Use visible text from Playwright (most reliable) or fall back to HTML extraction
        if visible_text and len(visible_text) > 100:
            logger.info(f"📄 Using Playwright visible text: {len(visible_text)} chars")
            text = visible_text
        elif html:
            soup = BeautifulSoup(html, "lxml")
            for elem in soup(["script", "style", "nav", "noscript", "svg", "iframe"]):
                elem.decompose()
            text = self._extract_markdown_text(soup)
            logger.info(f"📄 Using HTML extracted text: {len(text)} chars")
        else:
            text = ""

        # Prepend card markers if found
        if card_header and text:
            logger.info(f"📦 Card patterns found: {len(card_header)} chars + visible text")
            text = card_header + "\n\n--- VOLLTEXT ---\n\n" + text
        elif text:
            logger.info(f"📦 Text: {len(text)} chars")

        # Truncate if needed
        if len(text) > MAX_TEXT_EXTRACT:
            text = text[:MAX_TEXT_EXTRACT]
            logger.info(f"📦 Truncated to {MAX_TEXT_EXTRACT} chars")

        # Skip AI extraction if we have almost no text
        if len(text) < 100:
            logger.warning(f"⚠️ Not enough text for extraction ({len(text)} chars)")
            return []

        # Use AI to extract contacts
        logger.info(f"🤖 Running AI contact extraction on {len(text)} chars...")
        return await extract_contacts_from_page(
            text, company_name, "team",
            target_titles=target_titles,
            job_category=job_category,
            company_domain=company_domain
        )

    def _extract_structured_text(self, soup: BeautifulSoup) -> str:
        """
        Extract text from HTML with structure preservation.
        Full text is ALWAYS preserved — card markers are a bonus prefix.

        Strategy:
        1. Try to detect card patterns → ---PERSON--- markers as prefix
        2. Always extract full text with markdown formatting
        3. Fallback to plain get_text() if markdown extraction fails
        """
        # Stufe 1: Try card pattern detection as bonus header
        card_header = self._try_extract_cards(soup)

        # Stufe 2: Markdown-formatted full text (always)
        full_text = self._extract_markdown_text(soup)

        if not full_text or len(full_text) < 100:
            # Stufe 3: Fallback to plain text
            full_text = soup.get_text(separator="\n", strip=True)
            logger.info(f"📦 Fallback to plain text: {len(full_text)} chars")

        if card_header:
            logger.info(f"📦 Card patterns found + full text: {len(card_header)} + {len(full_text)} chars")
            return card_header + "\n\n--- VOLLTEXT ---\n\n" + full_text

        logger.info(f"📦 Markdown-formatted text: {len(full_text)} chars")
        return full_text

    def _try_extract_cards(self, soup: BeautifulSoup) -> str:
        """
        Try to detect repeating card/profile patterns in the HTML.
        Returns ---PERSON--- separated blocks if found, empty string otherwise.
        """
        CARD_SELECTORS = [
            # Specific team card classes
            ".team-member", ".team-card", ".person-card", ".employee-card",
            ".mitarbeiter", ".staff-member", ".member-card", ".ansprechpartner",
            # Grid/list children with team-related parents
            "[class*='team'] > div", "[class*='team'] > li",
            "[class*='people'] > div", "[class*='staff'] > div",
            "[class*='member'] > div", "[class*='mitarbeiter'] > div",
            # CMS-specific patterns
            ".elementor-team-member", ".wp-block-team > div",
            ".et_pb_team_member", ".avia-team-member",
            # Generic patterns (checked last, more cautious)
            "[class*='person'] > div", "[class*='profile'] > div",
        ]

        for selector in CARD_SELECTORS:
            try:
                cards = soup.select(selector)
            except Exception:
                continue

            if len(cards) < 2:
                continue

            blocks = []
            for card in cards:
                card_text = card.get_text(separator="\n", strip=True)
                # Valid person card: 10-500 chars (too short = empty, too long = not a card)
                if 10 < len(card_text) < 500:
                    blocks.append(f"---PERSON---\n{card_text}")

            if len(blocks) >= 2:
                logger.info(f"  ✓ Card pattern found: {selector} ({len(blocks)} cards)")
                return "\n\n".join(blocks)

        return ""

    def _extract_markdown_text(self, soup: BeautifulSoup) -> str:
        """
        Extract full text from HTML with light structure preservation.
        Uses get_text() for reliability but enhances with heading/bold markers.
        Always uses full body to avoid missing content in partial containers.
        """
        container = soup.find('body') or soup

        if not container:
            return ""

        # Also remove footer (usually has no team info, just legal text)
        for footer in container.find_all('footer'):
            footer.decompose()

        # Enhance HTML with markers before get_text()
        for heading in container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            level = int(heading.name[1])
            heading.insert_before(f"\n{'#' * level} ")
            heading.insert_after("\n")

        for bold in container.find_all(['strong', 'b']):
            bold.insert_before("**")
            bold.insert_after("**")

        for link in container.find_all('a', href=True):
            href = link.get('href', '')
            if href.startswith('mailto:'):
                email = href.replace('mailto:', '').split('?')[0]
                link.insert_after(f" ({email})")
            elif href.startswith('tel:'):
                phone = href.replace('tel:', '')
                link.insert_after(f" ({phone})")

        # Extract text with newline separator
        text = container.get_text(separator="\n", strip=True)

        # Clean up excessive blank lines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text.strip()

    async def _scrape_with_playwright_v2(self, url: str) -> Optional[Tuple[str, str]]:
        """
        Improved Playwright scraping for JS-heavy team pages.

        Improvements over v1:
        - Wait for team-specific selectors
        - Longer initial wait (8s instead of 4s)
        - Full page scroll (top → bottom → top)
        - Multiple scroll passes for lazy loading
        """
        try:
            from playwright.async_api import async_playwright

            logger.info(f"🎭 Starting Playwright (improved settings)...")

            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--disable-http2', '--disable-blink-features=AutomationControlled']
                )
                try:
                    context = await browser.new_context(
                        viewport={'width': 1280, 'height': 900},
                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    )
                    page = await context.new_page()

                    # Navigate with longer timeout
                    logger.info(f"  → Navigating to {url}")
                    try:
                        await page.goto(url, wait_until='domcontentloaded', timeout=20000)
                    except Exception as e:
                        logger.warning(f"  ⚠️ Navigation error: {e}")
                        return None

                    # Wait for network to settle
                    logger.info(f"  → Waiting for network idle...")
                    try:
                        await page.wait_for_load_state('networkidle', timeout=10000)
                    except Exception:
                        logger.debug(f"  → Network idle timeout (continuing)")

                    # Try to wait for team-specific selectors
                    logger.info(f"  → Looking for team content selectors...")
                    team_found = False
                    for selector in TEAM_PAGE_SELECTORS[:10]:  # Check first 10 selectors
                        try:
                            await page.wait_for_selector(selector, timeout=2000)
                            logger.info(f"  ✓ Found team selector: {selector}")
                            team_found = True
                            break
                        except Exception:
                            continue

                    if not team_found:
                        # No team selector found - wait longer for JS to render
                        logger.info(f"  → No team selectors found, waiting 8s for JS...")
                        await page.wait_for_timeout(8000)
                    else:
                        # Team selector found - small additional wait
                        await page.wait_for_timeout(2000)

                    # Try to dismiss cookie banners / popups
                    cookie_selectors = [
                        "[class*='cookie'] button", "[id*='cookie'] button",
                        ".cc-dismiss", "#accept-cookies", ".cookie-accept",
                        "[data-action='accept']", ".consent-accept",
                        "button[class*='accept']", "button[class*='agree']",
                    ]
                    for sel in cookie_selectors:
                        try:
                            btn = page.locator(sel).first
                            if await btn.is_visible(timeout=500):
                                await btn.click()
                                logger.debug(f"  → Dismissed cookie banner: {sel}")
                                break
                        except Exception:
                            pass

                    # Full page scroll to trigger lazy loading
                    logger.info(f"  → Scrolling page to load lazy content...")
                    await self._full_page_scroll(page)

                    # Get both HTML (for card detection) and visible text (reliable)
                    try:
                        html = await page.evaluate("document.body ? document.body.innerHTML : document.documentElement.outerHTML")
                    except Exception:
                        html = await page.content()

                    # Get visible text directly from Playwright (most reliable)
                    try:
                        visible_text = await page.inner_text('body')
                    except Exception:
                        visible_text = ""

                    logger.info(f"  ✓ Got {len(html)} bytes HTML, {len(visible_text)} chars visible text")

                    if len(html) > MAX_PAGE_SIZE:
                        html = html[:MAX_PAGE_SIZE]

                    # Return both as tuple (html, visible_text)
                    return (html, visible_text)

                finally:
                    await browser.close()

        except ImportError:
            logger.warning("⚠️ Playwright not installed")
            return None
        except Exception as e:
            logger.warning(f"❌ Playwright error: {e}")
            return None

    async def _full_page_scroll(self, page) -> None:
        """
        Scroll page fully to trigger all lazy loading.

        Pattern: Scroll down in steps, wait, then scroll back up.
        """
        try:
            # Get page height
            height = await page.evaluate("document.body.scrollHeight")
            viewport_height = 900

            # Scroll down in steps
            current = 0
            while current < height:
                current += viewport_height
                await page.evaluate(f"window.scrollTo(0, {current})")
                await page.wait_for_timeout(300)

            # Wait at bottom
            await page.wait_for_timeout(1000)

            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)

            # Final scroll to middle (where team often is)
            await page.evaluate(f"window.scrollTo(0, {height // 2})")
            await page.wait_for_timeout(500)

        except Exception as e:
            logger.debug(f"Scroll error: {e}")

    async def _scrape_with_httpx(self, url: str) -> Optional[str]:
        """Fallback scraping with httpx (no JS rendering)."""
        client = await self._get_client()

        try:
            response = await client.get(url, timeout=15)
            if response.status_code != 200:
                return None

            content = response.text
            if len(content) > MAX_PAGE_SIZE:
                content = content[:MAX_PAGE_SIZE]

            return content

        except Exception as e:
            logger.debug(f"httpx scrape failed: {e}")
            return None

    def _deduplicate_pages(self, pages: List[DiscoveredPage]) -> List[DiscoveredPage]:
        """Remove duplicate pages by URL."""
        seen = set()
        unique = []
        for page in pages:
            # Normalize URL
            url = page.url.rstrip('/')
            if url not in seen:
                seen.add(url)
                unique.append(page)
        return unique

    def _deduplicate_contacts(self, contacts: List[ExtractedContact]) -> List[ExtractedContact]:
        """Remove duplicate contacts by name."""
        seen_names = set()
        unique = []

        for contact in contacts:
            name_lower = contact.name.lower().strip()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique.append(contact)

        return unique

    async def close(self):
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()


# Convenience function (maintains backwards compatibility)
async def discover_team_contacts(
    company_name: str,
    domain: Optional[str] = None,
    job_category: Optional[str] = None,
    target_titles: Optional[List[str]] = None
) -> TeamDiscoveryResult:
    """
    Discover team contacts for a company.

    Uses the new "2-clicks" smart discovery approach.
    """
    discovery = TeamDiscovery()
    try:
        return await discovery.discover_and_extract(
            company_name=company_name,
            domain=domain,
            job_category=job_category,
            target_titles=target_titles
        )
    finally:
        await discovery.close()
