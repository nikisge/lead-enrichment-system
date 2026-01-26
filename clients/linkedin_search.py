import logging
import re
import httpx
from typing import Optional, List

from config import get_settings
from utils.cost_tracker import track_google

logger = logging.getLogger(__name__)


class LinkedInSearchClient:
    """
    Search for LinkedIn profiles using Google Custom Search API.
    Useful when Apollo is not available (free plan limitation).
    """

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.google_api_key
        self.cse_id = settings.google_cse_id
        self.timeout = settings.api_timeout

    async def find_linkedin_profile(
        self,
        name: str,
        company: Optional[str] = None,
        domain: Optional[str] = None
    ) -> Optional[str]:
        """
        Search Google for a person's LinkedIn profile.

        IMPORTANT: Only returns profiles where we can verify the company match.
        Name-only searches are disabled to avoid returning wrong profiles.

        Args:
            name: Full name of the person
            company: Company name (REQUIRED for reliable results)
            domain: Company domain (optional, improves accuracy)

        Returns:
            LinkedIn profile URL or None
        """
        if not self.api_key or not self.cse_id:
            logger.warning("Google API key or CSE ID not configured")
            return None

        if not company:
            logger.warning(f"LinkedIn search for '{name}' skipped - company name required for reliable results")
            return None

        # Try search strategies - all require company/domain matching
        strategies = []

        # Strategy 1: Name + Company (most specific)
        strategies.append((f'"{name}" "{company}" site:linkedin.com/in', True))

        # Strategy 2: Name + Company without quotes on company
        strategies.append((f'"{name}" {company} site:linkedin.com/in', True))

        # Strategy 3: Name + Domain
        if domain:
            strategies.append((f'"{name}" {domain} site:linkedin.com/in', True))

        # Strategy 4: Just name - ONLY with company validation in results
        # This is a fallback but requires company to appear in snippet/title
        strategies.append((f'"{name}" site:linkedin.com/in', False))

        # Track candidates that failed heuristic check for AI fallback
        ai_fallback_candidates = []

        for query, company_in_query in strategies:
            result = await self._search_google(query, name, company, require_company_match=not company_in_query)
            if result:
                # Final safety net: validate URL matches searched name
                if self._url_matches_name(result, name):
                    logger.info(f"LinkedIn profile verified for '{name}': {result}")
                    return result
                else:
                    # Heuristic failed - save for potential AI fallback
                    logger.info(f"Heuristic rejected URL '{result}' for '{name}' - saved for AI fallback")
                    ai_fallback_candidates.append((result, company_in_query))
                    continue  # Try next strategy

        # AI Fallback: Try AI matching for rejected candidates (max 2 to save costs)
        if ai_fallback_candidates:
            logger.info(f"Trying AI fallback for {len(ai_fallback_candidates)} candidates...")
            for url, had_company in ai_fallback_candidates[:2]:
                if await self._url_matches_name_ai(url, name, company):
                    logger.info(f"LinkedIn profile verified by AI for '{name}': {url}")
                    return url

        logger.info(f"No LinkedIn profile found for '{name}' after all strategies and AI fallback")
        return None

    async def _search_google(
        self,
        query: str,
        name: str,
        company: Optional[str] = None,
        require_company_match: bool = False
    ) -> Optional[str]:
        """
        Search Google for LinkedIn profile.

        Args:
            require_company_match: If True, the company name MUST appear in the
                snippet/title to accept the result. Used for name-only searches.
        """
        logger.info(f"LinkedIn search query: {query}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 5  # Get top 5 results
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                track_google("linkedin_profile_search")  # Track Google API call
                data = response.json()

                items = data.get("items", [])
                total_results = data.get("searchInformation", {}).get("totalResults", "0")
                logger.info(f"Google returned {len(items)} results (total: {total_results})")

                if not items:
                    return None  # Try next strategy

                # Parse name for matching
                name_parts = name.lower().split()
                first_name = name_parts[0] if name_parts else ""
                last_name = name_parts[-1] if len(name_parts) > 1 else ""

                # Find best LinkedIn profile match
                best_match = None
                best_score = 0

                for item in items:
                    link = item.get("link", "")
                    title = item.get("title", "").lower()
                    snippet = item.get("snippet", "").lower()

                    # Must be a LinkedIn profile URL
                    if not self._is_linkedin_profile_url(link):
                        continue

                    # Calculate match score
                    score = 0

                    # Check name in title/snippet
                    if first_name and first_name in title:
                        score += 2
                    if last_name and last_name in title:
                        score += 3  # Last name more important
                    if first_name and first_name in snippet:
                        score += 1
                    if last_name and last_name in snippet:
                        score += 1

                    # Check company in snippet (if provided)
                    company_found_in_result = False
                    if company:
                        company_lower = company.lower()
                        company_words = [w for w in company_lower.split() if len(w) > 3]
                        for word in company_words:
                            if word in snippet or word in title:
                                score += 2
                                company_found_in_result = True
                                break

                    # If this is a name-only search, REQUIRE company in results
                    if require_company_match and not company_found_in_result:
                        logger.debug(f"Skipping {link} - company not found in snippet/title (required)")
                        continue

                    # CRITICAL: Validate URL slug matches the searched name
                    # This prevents returning wrong profiles (e.g., "valentin-bachmann" for "Matthias Handschuh")
                    if not self._url_matches_name(link, name):
                        logger.info(f"Skipping {link} - URL slug doesn't match searched name '{name}'")
                        continue

                    if score > best_score:
                        best_score = score
                        best_match = link

                # Require minimum score
                # - With company in query: score >= 3 (at least last name match)
                # - Name-only with company validation: score >= 5 (name + company match)
                min_score = 5 if require_company_match else 3

                if best_match and best_score >= min_score:
                    # Final safety check: verify URL matches name
                    if not self._url_matches_name(best_match, name):
                        logger.warning(f"Final check failed: URL '{best_match}' doesn't match name '{name}'")
                        return None
                    logger.info(f"Found LinkedIn profile (score={best_score}, min={min_score}): {best_match}")
                    return self._normalize_linkedin_url(best_match)

                if best_match and best_score < min_score:
                    logger.info(f"Rejected LinkedIn profile (score={best_score} < min={min_score}): {best_match}")

                return None  # Try next strategy

            except httpx.HTTPStatusError as e:
                logger.error(f"Google API error: {e.response.status_code} - {e.response.text}")
                return None
            except Exception as e:
                logger.error(f"Google search failed: {e}")
                return None

    def _is_linkedin_profile_url(self, url: str) -> bool:
        """Check if URL is a LinkedIn profile (not company page)."""
        if not url:
            return False

        # Must contain linkedin.com/in/ for personal profiles
        return "linkedin.com/in/" in url.lower()

    def _normalize_linkedin_url(self, url: str) -> str:
        """Normalize LinkedIn URL to standard format."""
        # Remove query params and trailing slashes
        url = url.split("?")[0].rstrip("/")

        # Ensure https
        if url.startswith("http://"):
            url = url.replace("http://", "https://")

        # Remove locale prefixes like /de/
        url = re.sub(r'linkedin\.com/[a-z]{2}/in/', 'linkedin.com/in/', url)

        return url

    def _normalize_german_chars(self, text: str) -> str:
        """Normalize German umlauts and special characters for matching."""
        replacements = {
            'ä': 'a', 'ae': 'a',
            'ö': 'o', 'oe': 'o',
            'ü': 'u', 'ue': 'u',
            'ß': 'ss',
            'é': 'e', 'è': 'e', 'ê': 'e',
            'á': 'a', 'à': 'a', 'â': 'a',
            'ó': 'o', 'ò': 'o', 'ô': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u',
            'í': 'i', 'ì': 'i', 'î': 'i',
        }
        result = text.lower()
        for old, new in replacements.items():
            result = result.replace(old, new)
        return result

    def _url_matches_name(self, url: str, name: str) -> bool:
        """
        CRITICAL: Validate that LinkedIn URL slug matches the searched name.

        This prevents returning wrong profiles (e.g., returning "valentin-bachmann"
        when searching for "Matthias Handschuh").

        Handles:
        - German umlauts: "glaetzle" matches "Glätzle"
        - Abbreviations: "m-mueller" matches "Max Müller"
        - Surname-only slugs: "glaetzle" matches "Alexander Glätzle"

        Args:
            url: LinkedIn profile URL (e.g., "linkedin.com/in/valentin-bachmann")
            name: Searched person name (e.g., "Matthias Handschuh")

        Returns:
            True if URL slug matches the searched name
        """
        if not url or not name:
            return False

        # Extract slug from URL (e.g., "valentin-bachmann" from linkedin.com/in/valentin-bachmann)
        match = re.search(r'linkedin\.com/in/([^/?]+)', url.lower())
        if not match:
            return False

        slug = match.group(1)  # e.g., "valentin-bachmann" or "glaetzle"
        # Remove trailing numbers/IDs (e.g., "max-mueller-123abc" -> "max-mueller")
        slug_clean = re.sub(r'-[a-f0-9]{6,}$', '', slug)
        slug_normalized = slug_clean.replace("-", " ").replace("_", " ")

        # Normalize for German characters
        slug_norm = self._normalize_german_chars(slug_normalized)

        # Parse searched name and normalize
        name_normalized = self._normalize_german_chars(name)
        name_parts = name_normalized.split()
        if not name_parts:
            return False

        first_name = name_parts[0]
        last_name = name_parts[-1] if len(name_parts) > 1 else ""

        # Check if at least first OR last name is in the slug
        slug_words = slug_norm.split()

        # Last name match (most reliable)
        last_name_match = last_name and (
            last_name in slug_words or
            slug_norm.endswith(last_name) or
            last_name in slug_norm or
            # Also check if slug IS the last name (surname-only slugs like "glaetzle")
            slug_norm == last_name or
            slug_norm.startswith(last_name)
        )

        # First name match
        first_name_match = first_name and (
            first_name in slug_words or
            slug_norm.startswith(first_name) or
            first_name in slug_norm
        )

        # Require at least one match (preferably last name as it's more unique)
        if last_name_match:
            return True
        if first_name_match and len(first_name) >= 4:  # Short first names might false-positive
            return True

        logger.debug(f"URL slug mismatch: searched '{name}' (normalized: '{name_normalized}') but URL has '{slug}' (normalized: '{slug_norm}')")
        return False

    async def _url_matches_name_ai(self, url: str, name: str, company_name: Optional[str] = None) -> bool:
        """
        AI-based fallback for URL matching when heuristics fail.
        Use sparingly - only when heuristic check returns False but we suspect it might be correct.
        """
        try:
            from clients.ai_extractor import ai_match_linkedin_to_name

            # Extract slug
            match = re.search(r'linkedin\.com/in/([^/?]+)', url.lower())
            if not match:
                return False

            slug = match.group(1)

            result = await ai_match_linkedin_to_name(
                linkedin_slug=slug,
                person_name=name,
                company_name=company_name
            )

            if result.get("matches") and result.get("confidence") in ["high", "medium"]:
                logger.info(f"AI matched LinkedIn URL: '{slug}' -> '{name}' ({result.get('reason')})")
                return True

            return False
        except Exception as e:
            logger.warning(f"AI LinkedIn matching failed: {e}")
            return False

    async def find_decision_maker(
        self,
        company: str,
        domain: Optional[str] = None,
        titles: Optional[list] = None,
        job_category: Optional[str] = None
    ) -> Optional[dict]:
        """
        Search for a decision maker at a company when no contact name is known.
        Returns the BEST single candidate (verified if possible).

        For multiple candidates, use find_multiple_decision_makers() instead.
        """
        candidates = await self.find_multiple_decision_makers(
            company=company,
            domain=domain,
            job_category=job_category,
            max_candidates=1
        )
        return candidates[0] if candidates else None

    async def find_multiple_decision_makers(
        self,
        company: str,
        domain: Optional[str] = None,
        job_category: Optional[str] = None,
        max_candidates: int = 3
    ) -> List[dict]:
        """
        Search for MULTIPLE decision makers at a company.
        Returns up to max_candidates, prioritizing verified current employees.

        Priority:
        1. HR / Recruiting / Personal (best for job postings)
        2. Department head matching job category
        3. General executives (fallback)

        Args:
            company: Company name
            domain: Company domain (optional)
            job_category: Job category like "IT", "Sales", etc.
            max_candidates: Maximum number of candidates to return (default 3)

        Returns:
            List of dicts with 'name', 'title', 'linkedin_url', 'verified_current'
        """
        if not self.api_key or not self.cse_id:
            logger.warning("Google API key or CSE ID not configured for decision maker search")
            return []

        all_candidates = []
        seen_urls = set()

        # Search 1: HR/Recruiting (combined query)
        hr_query = "Personalleiter OR HR OR Recruiting OR Personal"
        candidates = await self._search_decision_maker_combined(company, hr_query, domain, return_all=True)
        for c in candidates:
            if c["linkedin_url"] not in seen_urls:
                seen_urls.add(c["linkedin_url"])
                all_candidates.append(c)

        # Search 2: Job-category specific (if category provided)
        if job_category:
            category_query = self._get_category_query(job_category)
            if category_query:
                candidates = await self._search_decision_maker_combined(company, category_query, domain, return_all=True)
                for c in candidates:
                    if c["linkedin_url"] not in seen_urls:
                        seen_urls.add(c["linkedin_url"])
                        all_candidates.append(c)

        # Search 3: Executive fallback
        exec_query = "Geschäftsführer OR CEO OR Inhaber OR Managing Director"
        candidates = await self._search_decision_maker_combined(company, exec_query, domain, return_all=True)
        for c in candidates:
            if c["linkedin_url"] not in seen_urls:
                seen_urls.add(c["linkedin_url"])
                all_candidates.append(c)

        # Sort: verified first, then by score
        all_candidates.sort(key=lambda x: (not x.get("verified_current", False), -x.get("score", 0)))

        # Return top candidates
        result = all_candidates[:max_candidates]
        logger.info(f"Found {len(result)} decision maker candidates for {company} (verified: {sum(1 for c in result if c.get('verified_current'))})")

        return result

    def _get_category_query(self, category: str) -> Optional[str]:
        """Get combined search query for job category."""
        category_lower = category.lower()

        category_queries = {
            "it": "CTO OR IT-Leiter OR Head of IT OR Tech Lead",
            "software": "CTO OR Head of Engineering OR Tech Lead",
            "tech": "CTO OR IT-Leiter OR Head of IT",
            "sales": "Vertriebsleiter OR Sales Director OR Head of Sales",
            "vertrieb": "Vertriebsleiter OR Sales Director",
            "marketing": "CMO OR Marketing-Leiter OR Head of Marketing",
            "finance": "CFO OR Finanzleiter OR Head of Finance",
            "finanzen": "CFO OR Finanzleiter",
            "operations": "COO OR Betriebsleiter OR Operations",
            "produktion": "Produktionsleiter OR Werkleiter",
            "logistik": "Logistikleiter OR Supply Chain",
            "einkauf": "Einkaufsleiter OR Procurement",
            "consulting": "Partner OR Principal OR Director",
            "beratung": "Partner OR Principal OR Managing Consultant",
            "healthcare": "Chefarzt OR Medical Director OR Klinikleiter",
            "medizin": "Chefarzt OR Ärztlicher Direktor",
        }

        for key, query in category_queries.items():
            if key in category_lower:
                return query

        return None

    async def _search_decision_maker_combined(
        self,
        company: str,
        title_query: str,
        domain: Optional[str] = None,
        return_all: bool = False
    ) -> List[dict]:
        """
        Search Google with combined title query (OR syntax).

        Args:
            return_all: If True, return all matching candidates. If False, return only best one.

        Returns:
            List of candidate dicts (empty list if none found)
        """
        # Build search query with OR combinations
        query = f'({title_query}) "{company}" site:linkedin.com/in'

        logger.info(f"Decision maker search: {query[:80]}...")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 10  # Get more results to find best match
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                track_google("decision_maker_search")  # Track Google API call
                data = response.json()

                items = data.get("items", [])
                if not items:
                    return []

                # Find matches from results - prioritize verified current employees
                candidates = []

                for item in items:
                    link = item.get("link", "")
                    item_title = item.get("title", "")
                    snippet = item.get("snippet", "")

                    # Must be a LinkedIn profile
                    if not self._is_linkedin_profile_url(link):
                        continue

                    # Check if company name appears
                    company_lower = company.lower()
                    company_words = [w for w in company_lower.split() if len(w) > 2]

                    combined_text = (item_title + " " + snippet).lower()
                    company_match = any(word in combined_text for word in company_words)

                    if not company_match:
                        continue

                    # Extract name from LinkedIn title
                    name = self._extract_name_from_linkedin_title(item_title)
                    if not name:
                        continue

                    linkedin_url = self._normalize_linkedin_url(link)
                    extracted_title = self._extract_title_from_snippet(snippet, "")

                    # Check if person is CURRENTLY at this company
                    is_current = self._is_currently_at_company(snippet, item_title, company)

                    candidates.append({
                        "name": name,
                        "title": extracted_title,
                        "linkedin_url": linkedin_url,
                        "verified_current": is_current,
                        "score": 10 if is_current else 1  # Prioritize current employees
                    })

                if not candidates:
                    return []

                # Sort by score (verified current employees first)
                candidates.sort(key=lambda x: x["score"], reverse=True)

                if return_all:
                    return candidates

                # Return only best one
                best = candidates[0]
                if best["verified_current"]:
                    logger.info(f"Found VERIFIED: {best['name']} ({best['title']}) aktuell bei {company}")
                else:
                    logger.info(f"Found UNVERIFIED: {best['name']} ({best['title']}) - könnte nicht mehr bei {company} sein")

                return [best]

            except httpx.HTTPStatusError as e:
                logger.error(f"Google API error: {e.response.status_code}")
                return []
            except Exception as e:
                logger.error(f"Decision maker search failed: {e}")
                return []

    def _is_currently_at_company(self, snippet: str, title: str, company: str) -> bool:
        """
        Check if LinkedIn snippet/title indicates person is CURRENTLY at the company.
        Returns True if there's strong evidence they currently work there.
        """
        text = (snippet + " " + title).lower()
        company_lower = company.lower()

        # Strong indicators of current employment
        current_indicators = [
            # German
            f"aktuell bei {company_lower}",
            f"aktuell: {company_lower}",
            f"derzeit bei {company_lower}",
            f"derzeit: {company_lower}",
            f"tätig bei {company_lower}",
            # English
            f"currently at {company_lower}",
            f"current: {company_lower}",
            f"working at {company_lower}",
            # LinkedIn patterns - title usually shows current position
            f"bei {company_lower}",
            f"at {company_lower}",
        ]

        # Check for current indicators with company name
        for indicator in current_indicators:
            if indicator in text:
                return True

        # Check if company appears right after title patterns (LinkedIn format)
        # e.g. "Max Müller - HR Manager bei Firma XYZ"
        company_words = [w for w in company_lower.split() if len(w) > 2]
        for word in company_words:
            # Pattern: "titel bei/at firma" suggests current role
            if re.search(rf'(manager|leiter|director|head|ceo|cto|cfo)\s+(bei|at|@)\s+\w*{word}', text):
                return True

        # Check for negative indicators (former employment)
        former_indicators = [
            "ehemalig", "ehemals", "former", "previously", "ex-",
            "bis 20", "until 20", "left in", "verließ"
        ]

        for indicator in former_indicators:
            if indicator in text:
                return False

        # If company is in the text near the beginning (typical for current job), consider it likely current
        # LinkedIn snippets usually start with current position
        first_100_chars = text[:100]
        if any(word in first_100_chars for word in company_words):
            return True

        return False

    def _get_category_titles(self, category: str) -> list:
        """Get relevant department head titles based on job category."""
        category_lower = category.lower()

        category_map = {
            "it": ["IT-Leiter", "Head of IT", "CTO", "Tech Lead", "IT Director", "Head of Engineering"],
            "software": ["CTO", "Head of Engineering", "Tech Lead", "VP Engineering", "IT-Leiter"],
            "tech": ["CTO", "Head of IT", "Tech Lead", "IT-Leiter", "Head of Engineering"],
            "sales": ["Vertriebsleiter", "Sales Director", "Head of Sales", "VP Sales", "CSO"],
            "vertrieb": ["Vertriebsleiter", "Sales Director", "Head of Sales", "Verkaufsleiter"],
            "marketing": ["Marketing-Leiter", "Head of Marketing", "CMO", "Marketing Director"],
            "finance": ["CFO", "Finanzleiter", "Head of Finance", "Finance Director"],
            "finanzen": ["CFO", "Finanzleiter", "Head of Finance", "Kaufmännischer Leiter"],
            "operations": ["COO", "Operations Manager", "Betriebsleiter", "Head of Operations"],
            "produktion": ["Produktionsleiter", "Head of Production", "Werkleiter", "COO"],
            "logistik": ["Logistikleiter", "Head of Logistics", "Supply Chain Manager"],
            "einkauf": ["Einkaufsleiter", "Head of Procurement", "Purchasing Manager"],
            "personal": ["Personalleiter", "HR Manager", "Head of HR", "HR Director"],
            "hr": ["Personalleiter", "HR Manager", "Head of HR", "HR Director"],
            "consulting": ["Partner", "Managing Consultant", "Principal", "Director"],
            "beratung": ["Partner", "Managing Consultant", "Principal", "Geschäftsführer"],
            "healthcare": ["Klinikleiter", "Chefarzt", "Medical Director", "Geschäftsführer"],
            "medizin": ["Klinikleiter", "Chefarzt", "Medical Director", "Ärztlicher Direktor"],
        }

        # Find matching category
        for key, titles in category_map.items():
            if key in category_lower:
                return titles

        # Default: general management
        return ["Abteilungsleiter", "Team Lead", "Head of", "Manager"]

    async def _search_decision_maker_google(
        self,
        company: str,
        title: str,
        domain: Optional[str] = None
    ) -> Optional[dict]:
        """Search Google for a specific title at a company."""
        # Build search query
        query = f'"{title}" "{company}" site:linkedin.com/in'
        if domain:
            query = f'"{title}" "{company}" OR "{domain}" site:linkedin.com/in'

        logger.info(f"Decision maker search: {query}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            url = "https://www.googleapis.com/customsearch/v1"

            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": 5
            }

            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                track_google("decision_maker_linkedin_search")  # Track Google API call
                data = response.json()

                items = data.get("items", [])
                if not items:
                    return None

                # Find the best match
                for item in items:
                    link = item.get("link", "")
                    item_title = item.get("title", "")
                    snippet = item.get("snippet", "")

                    # Must be a LinkedIn profile
                    if not self._is_linkedin_profile_url(link):
                        continue

                    # Check if company name appears in title or snippet
                    company_lower = company.lower()
                    company_words = [w for w in company_lower.split() if len(w) > 2]

                    company_match = any(
                        word in item_title.lower() or word in snippet.lower()
                        for word in company_words
                    )

                    if not company_match:
                        continue

                    # Extract name from LinkedIn title (usually "Name - Title | LinkedIn")
                    name = self._extract_name_from_linkedin_title(item_title)
                    if not name:
                        continue

                    linkedin_url = self._normalize_linkedin_url(link)

                    # Try to extract actual title from snippet
                    extracted_title = self._extract_title_from_snippet(snippet, title)

                    logger.info(f"Found decision maker: {name} ({extracted_title}) at {company}")
                    return {
                        "name": name,
                        "title": extracted_title,
                        "linkedin_url": linkedin_url
                    }

                return None

            except httpx.HTTPStatusError as e:
                logger.error(f"Google API error for decision maker search: {e.response.status_code}")
                return None
            except Exception as e:
                logger.error(f"Decision maker search failed: {e}")
                return None

    def _extract_name_from_linkedin_title(self, title: str) -> Optional[str]:
        """Extract person name from LinkedIn search result title."""
        # LinkedIn titles are usually "FirstName LastName - Title | LinkedIn"
        # or "FirstName LastName | LinkedIn"
        if not title:
            return None

        # Remove " | LinkedIn" or " - LinkedIn" suffix
        title = re.sub(r'\s*[\|\-]\s*LinkedIn.*$', '', title, flags=re.IGNORECASE)

        # Split by " - " or " – " to separate name from job title
        parts = re.split(r'\s*[\-–]\s*', title)
        if parts:
            name = parts[0].strip()
            # Basic validation: should have at least 2 words
            if len(name.split()) >= 2:
                return name

        return None

    def _extract_title_from_snippet(self, snippet: str, searched_title: str) -> str:
        """Try to extract actual job title from snippet, fallback to searched title."""
        # Common patterns in LinkedIn snippets
        patterns = [
            r'(Geschäftsführer(?:in)?)',
            r'(CEO)',
            r'(CTO)',
            r'(COO)',
            r'(CFO)',
            r'(Managing Director)',
            r'(Director\s+\w+)',
            r'(Head of\s+\w+)',
            r'(Leiter(?:in)?\s+\w+)',
            r'(VP\s+\w+)',
            r'(Founder)',
            r'(Inhaber(?:in)?)',
            r'(Owner)',
        ]

        for pattern in patterns:
            match = re.search(pattern, snippet, re.IGNORECASE)
            if match:
                return match.group(1)

        return searched_title


async def search_linkedin(
    name: str,
    company: Optional[str] = None,
    domain: Optional[str] = None
) -> Optional[str]:
    """
    Convenience function to search for LinkedIn profile.
    """
    client = LinkedInSearchClient()
    return await client.find_linkedin_profile(name, company, domain)


async def search_decision_maker(
    company: str,
    domain: Optional[str] = None,
    titles: Optional[list] = None
) -> Optional[dict]:
    """
    Search for a decision maker at a company when no contact name is known.
    Uses Google to find LinkedIn profiles of executives/managers.

    Returns:
        dict with 'name', 'title', 'linkedin_url' or None
    """
    client = LinkedInSearchClient()
    return await client.find_decision_maker(company, domain, titles)
