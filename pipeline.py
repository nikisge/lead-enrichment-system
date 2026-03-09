"""
Lead Enrichment Pipeline v4 - KI-basiert

Optimierter Flow mit intelligenter Kontakt-Erkennung:
1. Job Posting parsen (Claude Sonnet)
2. Parallel: Job URL, Impressum, Team Discovery scrapen
3. KI-basierte Extraktion und Validierung
4. LinkedIn-Suche für validierte Kandidaten
5. Phone Enrichment (FullEnrich, Kaspr)
6. Company Research

Ersetzt regelbasierte Checks durch kontextabhängige KI-Analyse.
"""

import logging
import re
import asyncio
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse

import httpx

from config import get_settings
from models import (
    WebhookPayload, EnrichmentResult, CompanyInfo, CompanyIntel,
    DecisionMaker, PhoneResult, PhoneSource, PhoneType, PhoneStatus
)
from llm_parser import parse_job_posting, get_last_parse_warnings
from clients.kaspr import get_kaspr_client
from clients.fullenrich import get_fullenrich_client
from clients.bettercontact import get_bettercontact_client
from clients.impressum import ImpressumScraper
from clients.linkedin_search import LinkedInSearchClient
from clients.company_research import CompanyResearcher
from clients.job_scraper import JobUrlScraper
from clients.apify_linkedin import get_apify_linkedin_client

# New AI-based modules
from clients.llm_client import get_llm_client
from clients.ai_extractor import (
    extract_job_posting_contact,
    extract_impressum_data,
    ExtractedContact,
    ai_match_email_to_person,
    ai_match_linkedin_to_name
)
from clients.ai_validator import (
    validate_and_rank_candidates,
    validate_linkedin_match,
    CandidateValidation
)
from clients.team_discovery import discover_team_contacts, TeamDiscoveryResult

from utils.phone import normalize_phone_number
from utils.stats import track_phone_attempt, track_pipeline_result
from utils.cost_tracker import (
    start_cost_tracking, log_cost_summary,
    track_llm, track_openrouter, track_google,
    track_enrichment, track_apify
)

logger = logging.getLogger(__name__)

# Domains to skip in search results (job portals, social media, directories, etc.)
# Shared across DDG, Serper, Google CSE domain discovery
SKIP_DOMAINS = frozenset({
    # Social / Professional Networks
    'linkedin.com', 'xing.com', 'facebook.com', 'twitter.com',
    'instagram.com', 'youtube.com', 'wikipedia.org',
    # Job Portals (DE/AT/CH)
    'kununu.de', 'glassdoor.com', 'glassdoor.de',
    'indeed.com', 'indeed.de', 'stepstone.de', 'stepstone.at', 'stepstone.ch',
    'monster.de', 'monster.at', 'monster.ch',
    'karriere.at', 'jobs.ch', 'jobware.de', 'stellenanzeigen.de',
    'gehalt.de', 'hokify.de', 'hokify.at', 'meinestadt.de', 'arbeitsagentur.de',
    # Business Directories / Data
    'dnb.com', 'creditreform.de', 'northdata.com', 'webvalid.de',
    'opencorpdata.com', 'cylex.de', 'firmenwissen.de', 'unternehmensregister.de',
    'northdata.de', 'wlw.de', 'kompany.de',
    'handelsregister.de', 'bundesanzeiger.de',
    # Search Engines
    'google.com', 'google.de',
    # Misc Directories
    'implisense.com', '11880.com', 'zaubee.com', 'sortlist.com',
    'sortlist.de', 'freelancermap.de', 'wer-zu-wem.de',
    'unternehmen24.info', 'unternehmensverzeichnis.org',
    'firmenabc.at', 'herold.at', 'zefix.ch',
    # ATS / Recruiting Platforms
    'join.com', 'onlyfy.com', 'breezy.hr', 'bamboohr.com', 'ashbyhq.com',
    'dvinci.de', 'coveto.de', 'd.vinci.de',
    'personio.de', 'softgarden.de', 'recruitingapp.com',
    'workday.com', 'greenhouse.io', 'lever.co', 'recruitee.com',
    'icims.com', 'taleo.net', 'successfactors.com',
    'smartrecruiters.com', 'jobvite.com', 'workable.com',
})


def _domain_relevance_score(domain: str, company_name: str) -> int:
    """Score how relevant a domain is for a company name.
    Higher score = more relevant. Handles umlauts in both directions."""
    name_lower = company_name.lower()
    for suffix in [' gmbh & co. kgaa', ' gmbh & co. kg', ' gmbh & co kg',
                  ' partg mbb', ' partg', ' gmbh', ' ggmbh', ' ag', ' kgaa',
                  ' kg', ' ohg', ' mbh', ' ug', ' gbr', ' eg', ' e.v.', ' co.', ' & co']:
        name_lower = name_lower.replace(suffix, '')
    name_words = [w for w in name_lower.split() if len(w) >= 3]
    domain_base = domain.split('.')[0].lower()

    score = 0
    for word in name_words:
        # Direct match
        if word in domain_base:
            score += 10
        # Umlaut -> ASCII (ö -> oe)
        word_converted = word
        for umlaut, replacement in [('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss')]:
            word_converted = word_converted.replace(umlaut, replacement)
        if word_converted != word and word_converted in domain_base:
            score += 10
        # Reverse: ASCII -> Umlaut (domain has 'ae', word has 'ä')
        domain_unconverted = domain_base
        for replacement, umlaut in [('ae', 'ä'), ('oe', 'ö'), ('ue', 'ü')]:
            domain_unconverted = domain_unconverted.replace(replacement, umlaut)
        if word in domain_unconverted and domain_unconverted != domain_base:
            score += 10
    return score


def _safe_first_name(name: str) -> str:
    """Extract first name safely, return 'unknown' if empty."""
    parts = (name or "").split()
    return parts[0] if parts else "unknown"


def _normalize_name_for_dedup(name: str) -> str:
    """Normalize name for deduplication: lowercase + umlaut normalization."""
    result = (name or "").lower()
    result = result.replace('ü', 'ue').replace('ä', 'ae').replace('ö', 'oe').replace('ß', 'ss')
    return result


def _is_valid_dach_phone(number: str) -> bool:
    """
    Check if phone number is a valid DACH phone number.
    Accepts: +49, +43, +41, 0049, 0043, 0041, 0xxx (German domestic)

    Validates:
    - Min 7 digits (short landlines like 089 21760)
    - Max 15 digits (international standard)
    - Filters service numbers (0800, 0900, 0180, 0137, 0700)
    """
    if not number:
        return False

    cleaned = re.sub(r'[^\d+]', '', number)
    digits_only = re.sub(r'\D', '', cleaned)

    # Length validation: min 7 (short landline like 089 21760), max 15 digits
    if len(digits_only) < 7 or len(digits_only) > 15:
        logger.debug(f"Invalid phone length: {number} ({len(digits_only)} digits)")
        return False

    # Filter German service numbers (useless for sales outreach)
    # Normalize to domestic format for service number check
    domestic = cleaned
    if domestic.startswith('+49'):
        domestic = '0' + domestic[3:]
    elif domestic.startswith('0049'):
        domestic = '0' + domestic[4:]

    service_prefixes = ('0800', '0900', '0180', '0137', '0700', '0190', '0191')
    if domestic.startswith(service_prefixes):
        logger.debug(f"Filtered service number: {number}")
        return False

    # Valid DACH prefixes
    if cleaned.startswith(('+49', '+43', '+41')):
        return True
    if cleaned.startswith(('0049', '0043', '0041')):
        return True
    if cleaned.startswith('0') and not cleaned.startswith('00'):
        return True

    return False


def _extract_phone_from_html(html: str) -> Optional[str]:
    """
    Extract a DACH company phone number from HTML content.
    Checks tel: links, Schema.org telephone, and common DACH phone patterns.
    Returns the first valid DACH phone number found, or None.
    """
    if not html:
        return None

    # 1. tel: links (most reliable - explicit phone markup)
    tel_pattern = re.compile(r'href=["\']tel:([+\d\s\-/().]+)["\']', re.IGNORECASE)
    for match in tel_pattern.finditer(html):
        number = match.group(1).strip()
        if _is_valid_dach_phone(number):
            return number

    # 2. Schema.org "telephone" property (structured data)
    schema_pattern = re.compile(r'"telephone"\s*:\s*"([^"]+)"', re.IGNORECASE)
    for match in schema_pattern.finditer(html):
        number = match.group(1).strip()
        if _is_valid_dach_phone(number):
            return number

    # 3. Common DACH phone patterns in visible text
    # Strip HTML tags first for cleaner matching
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text)

    dach_patterns = [
        # +49 (0) 123 456-789, +49 123 456789, etc.
        re.compile(r'(\+49[\s\-/().\d]{8,20})'),
        re.compile(r'(\+43[\s\-/().\d]{8,20})'),
        re.compile(r'(\+41[\s\-/().\d]{8,20})'),
        # 0049 format
        re.compile(r'(0049[\s\-/().\d]{8,18})'),
        re.compile(r'(0043[\s\-/().\d]{8,18})'),
        re.compile(r'(0041[\s\-/().\d]{8,18})'),
    ]

    for pattern in dach_patterns:
        for match in pattern.finditer(text):
            number = match.group(1).strip()
            # Clean up trailing punctuation
            number = re.sub(r'[\s\-/().]+$', '', number)
            if _is_valid_dach_phone(number):
                return number

    return None


async def _extract_phone_from_homepage(domain: str) -> Optional[str]:
    """
    Fetch homepage HTML and extract company phone number.
    Quick check with 5s timeout - runs before parallel scraping.
    """
    try:
        async with httpx.AsyncClient(
            timeout=5.0,
            follow_redirects=True,
            verify=False,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        ) as client:
            response = await client.get(
                f"https://{domain}",
                follow_redirects=True,
                headers={"Accept": "text/html"}
            )
            if response.status_code < 400 and response.text:
                phone = _extract_phone_from_html(response.text[:20000])
                if phone:
                    logger.info(f"Homepage phone found for {domain}: {phone[:8]}...")
                    return phone
    except Exception as e:
        logger.debug(f"Homepage phone extraction for {domain} failed: {e}")
    return None


PIPELINE_TIMEOUT_SECONDS = 480  # 8 min - waterfall services need time for all providers


async def enrich_lead(
    payload: WebhookPayload,
    skip_paid_apis: bool = False
) -> EnrichmentResult:
    """
    Main enrichment pipeline - KI-basierter Flow v4.
    Wrapped with a timeout to ensure we return before n8n times out.
    """
    _bg_tasks: list[asyncio.Task] = []
    try:
        return await asyncio.wait_for(
            _enrich_lead_inner(payload, skip_paid_apis, _bg_tasks=_bg_tasks),
            timeout=PIPELINE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error(f"Pipeline timeout ({PIPELINE_TIMEOUT_SECONDS}s) for {payload.company}")
        for task in _bg_tasks:
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled background task on timeout: {task.get_name()}")
        return EnrichmentResult(
            success=False,
            company=CompanyInfo(name=payload.company),
            phone_status=PhoneStatus.API_NO_RESULT,
            emails=[],
            enrichment_path=["pipeline_timeout"],
            warnings=[f"Pipeline timeout nach {PIPELINE_TIMEOUT_SECONDS}s"],
            job_id=payload.id,
            job_title=payload.title
        )


async def _enrich_lead_inner(
    payload: WebhookPayload,
    skip_paid_apis: bool = False,
    _bg_tasks: Optional[list] = None,
) -> EnrichmentResult:
    """
    Inner pipeline logic.

    Flow:
    1.  LLM Parse -> Extract contact name, company, email from job posting
    2.  Google Domain Search -> If no domain, find via Google (FREE)
    3.  PARALLEL SCRAPING:
        - Job URL scraping with AI extraction
        - Impressum scraping with AI extraction
        - Team Page Discovery (Google + AI)
    4.  KI-Validierung & Ranking aller Kandidaten
    5.  LinkedIn Search für validierte Kandidaten
    6.  Phone Enrichment (FullEnrich, Kaspr)
    7.  Company Research + Sales Brief

    Args:
        payload: Job posting data
        skip_paid_apis: If True, skip all paid API calls (for testing)
    """
    enrichment_path = []
    collected_emails: List[str] = []
    operational_alerts: Dict[str, bool] = {}

    # Check critical API keys at startup
    settings = get_settings()
    if not settings.anthropic_api_key:
        operational_alerts["anthropic_key_missing"] = True
        logger.warning("ALERT: Anthropic API key is missing!")
    if not settings.openrouter_api_key:
        operational_alerts["openrouter_key_missing"] = True
        logger.warning("ALERT: OpenRouter API key is missing!")

    # Start cost tracking for this enrichment run
    cost_tracker = start_cost_tracking(payload.company)

    # ========== PHASE 1: INITIALE DATENSAMMLUNG ==========

    logger.info(f"=== Starting enrichment for: {payload.company} ===")
    enrichment_path.append("llm_parse")

    # Step 1: Parse job posting with Claude
    parsed = await parse_job_posting(payload)
    track_llm("job_parse", tier="sonnet")  # Job parsing uses Sonnet
    # Log without sensitive data (phone numbers)
    logger.info(f"LLM extracted: domain={parsed.company_domain}, contact={parsed.contact_name}, has_phone={bool(parsed.contact_phone)}")

    # Create company info
    company_info = CompanyInfo(
        name=parsed.company_name,
        domain=parsed.company_domain,
        location=parsed.location or payload.location
    )

    # ========== DOMAIN DISCOVERY (Priority Order) ==========
    #
    # 1. Job URL Domain (MOST RELIABLE - if job is on company website)
    # 2. LLM extracted domain (from job description text)
    # 3. Serper.dev Google SERP ($0.001/query - real Google results)
    # 4. Google Knowledge Graph (FREE - 100K/day, official website URLs)
    # 5. DuckDuckGo Search (FREE - web search fallback)
    # 6. Heuristic (FREE - generates domains from company name)
    # 7. Google CSE (Last resort - uses API quota, 100 free/day)

    # Build job context for smarter AI domain validation
    job_context_for_validation = ""
    if payload.title:
        job_context_for_validation = f"Jobtitel: {payload.title}"
    if payload.category:
        job_context_for_validation += f" | Kategorie: {payload.category}"
    if payload.description:
        desc_snippet = payload.description[:200].replace('\n', ' ').strip()
        job_context_for_validation += f" | {desc_snippet}"

    # Priority 1: Extract domain from Job URL (FREE, MOST RELIABLE)
    if not company_info.domain and payload.url:
        url_domain = _extract_domain_from_job_url(payload.url, parsed.company_name)
        if url_domain:
            # Validate that this domain actually belongs to the company
            logger.info(f"Job URL domain found: {url_domain} - validating...")
            is_valid, ai_reason = await _ai_validate_domain(url_domain, parsed.company_name, job_context=job_context_for_validation)
            if is_valid:
                company_info.domain = url_domain
                company_info.website = f"https://{url_domain}"
                enrichment_path.append(f"domain:job_url->{url_domain}")
                logger.info(f"✓ Using domain from job URL: {url_domain}")
            else:
                logger.warning(f"Job URL domain {url_domain} rejected by AI validation ({ai_reason})")

    # Priority 2: Validate LLM-extracted domain
    if company_info.domain and parsed.company_name and not any(p.startswith("domain:job_url->") for p in enrichment_path):
        # LLM found a domain - validate it with AI
        logger.info(f"Validating LLM domain '{company_info.domain}' for '{parsed.company_name}'...")
        is_valid, ai_reason = await _ai_validate_domain(company_info.domain, parsed.company_name, job_context=job_context_for_validation)

        if not is_valid:
            # Domain doesn't match company - search for alternatives
            logger.warning(f"LLM domain '{company_info.domain}' rejected - searching alternatives...")
            enrichment_path.append(f"domain:llm_rejected->{company_info.domain} (AI: {ai_reason})")
            company_info.domain = None  # Reset - will search below
        else:
            enrichment_path.append(f"domain:llm_validated->{company_info.domain}")
            if not company_info.website:
                company_info.website = f"https://{company_info.domain}"

    # Priority 3-7: Serper → DuckDuckGo → Knowledge Graph → Google CSE → Heuristic
    if not company_info.domain and parsed.company_name:
        location = parsed.location or payload.location or ""

        # Strategy 3: Serper.dev (Places + Search parallel, AI selects best result, $0.002/query)
        if not company_info.domain and settings.serper_api_key:
            logger.info("No valid domain - trying Serper.dev (Places + Search)...")
            try:
                serper_result = await _serper_find_domain(
                    company_name=parsed.company_name,
                    job_context=job_context_for_validation,
                    location=location,
                )
                if serper_result:
                    company_info.domain, company_info.website, serper_phone, serper_address = serper_result
                    enrichment_path.append(f"domain:serper->{company_info.domain}")
                    logger.info(f"✓ Serper found domain: {company_info.domain}")
                    # Apply phone from Serper (Places/KG/Snippet)
                    if serper_phone and not company_info.phone and _is_valid_dach_phone(serper_phone):
                        company_info.phone = serper_phone
                        enrichment_path.append(f"company_phone:serper->{serper_phone[:8]}...")
                        logger.info(f"Company phone from Serper: {serper_phone[:8]}...")
                    # Apply address from Serper
                    if serper_address and not company_info.address:
                        company_info.address = serper_address
                        if not company_info.location:
                            company_info.location = serper_address
                        enrichment_path.append("company_address:serper")
                        logger.info(f"Company address from Serper: {serper_address[:40]}...")
            except Exception as e:
                logger.warning(f"Serper search error: {e}")
                enrichment_path.append("serper_error")

        # Strategy 4: DuckDuckGo search (FREE, ~1-2s)
        if not company_info.domain:
            logger.info("Trying DuckDuckGo search...")
            ddg_result = None
            try:
                ddg_result = await _duckduckgo_find_domain(
                    company_name=parsed.company_name,
                    job_title=payload.title or "",
                    job_context=job_context_for_validation,
                    location=location,
                )
            except Exception as e:
                logger.warning(f"DuckDuckGo search error: {e}")
                operational_alerts["ddg_search_error"] = True
                enrichment_path.append("ddg_error")
            if ddg_result:
                company_info.domain, company_info.website = ddg_result
                enrichment_path.append(f"domain:ddg->{company_info.domain}")
                logger.info(f"✓ DuckDuckGo found domain: {company_info.domain}")

        # Strategy 5: Google Knowledge Graph (FREE, 100K/day)
        if not company_info.domain and settings.google_api_key:
            logger.info("Trying Google Knowledge Graph...")
            try:
                kg_result = await _knowledge_graph_find_domain(
                    company_name=parsed.company_name,
                    job_context=job_context_for_validation,
                )
                if kg_result:
                    company_info.domain, company_info.website = kg_result
                    enrichment_path.append(f"domain:kg->{company_info.domain}")
                    logger.info(f"✓ Knowledge Graph found domain: {company_info.domain}")
            except Exception as e:
                logger.warning(f"Knowledge Graph error: {e}")
                enrichment_path.append("kg_error")

        # Strategy 6: Google CSE (100 free/day)
        if not company_info.domain:
            logger.info("Falling back to Google CSE...")
            found_domain = await _google_find_domain(parsed.company_name, validate_with_ai=True)
            if found_domain:
                company_info.domain = found_domain
                company_info.website = f"https://{company_info.domain}"
                enrichment_path.append(f"domain:google_cse->{company_info.domain}")

        # Strategy 7: Heuristic (FREE, last resort - generates domains from company name)
        if not company_info.domain:
            logger.info("Trying heuristic domain search...")
            heuristic_result = await _heuristic_find_domain(parsed.company_name, job_context=job_context_for_validation)
            if heuristic_result:
                company_info.domain, company_info.website = heuristic_result
                enrichment_path.append(f"domain:heuristic->{company_info.domain}")
                logger.info(f"✓ Heuristic found domain: {company_info.domain}")

        if not company_info.domain:
            logger.warning(f"No valid domain found for '{parsed.company_name}'")
            enrichment_path.append("domain:NONE_FOUND")

    # Normalize: extract main domain from any subdomain result
    if company_info.domain:
        main_domain = _extract_main_domain(company_info.domain)
        if main_domain and main_domain != company_info.domain:
            enrichment_path.append(f"subdomain_extracted:{company_info.domain}->{main_domain}")
            logger.info(f"Subdomain normalized: {company_info.domain} -> {main_domain}")
            company_info.domain = main_domain
            company_info.website = f"https://{main_domain}"

    # Ensure website URL is always set when we have a domain
    if company_info.domain and not company_info.website:
        company_info.website = f"https://{company_info.domain}"

    # ========== HOMEPAGE PHONE EXTRACTION ==========
    # Try to extract company phone from homepage before parallel scraping
    if company_info.domain and not company_info.phone:
        try:
            homepage_phone = await _extract_phone_from_homepage(company_info.domain)
            if homepage_phone:
                company_info.phone = homepage_phone
                enrichment_path.append(f"company_phone:homepage->{homepage_phone[:8]}...")
                logger.info(f"Company phone from homepage: {homepage_phone[:8]}...")
        except Exception as e:
            logger.debug(f"Homepage phone extraction failed: {e}")

    # ========== PHASE 2: PARALLEL SCRAPING ==========

    logger.info("Starting parallel scraping...")

    # Create tasks for parallel execution
    job_contact_task = _scrape_job_url_with_ai(
        url=payload.url,
        company_name=parsed.company_name,
        job_title=payload.title
    )

    impressum_task = _scrape_impressum_with_ai(
        domain=company_info.domain,
        company_name=parsed.company_name
    )

    team_discovery_task = discover_team_contacts(
        company_name=parsed.company_name,
        domain=company_info.domain,
        job_category=payload.category
    )

    # Execute in parallel
    job_contact, impressum_result, team_result = await asyncio.gather(
        job_contact_task,
        impressum_task,
        team_discovery_task,
        return_exceptions=True
    )

    # Handle exceptions from parallel tasks
    if isinstance(job_contact, Exception):
        logger.warning(f"Job URL scraping failed: {job_contact}")
        job_contact = None
        enrichment_path.append("job_url_error")
    elif job_contact:
        enrichment_path.append("job_url_ai_extracted")
        logger.info(f"Job URL extracted: name={job_contact.name}, email={job_contact.email}, phone={'yes' if job_contact.phone else 'no'}")

    if isinstance(impressum_result, Exception):
        logger.warning(f"Impressum scraping failed: {impressum_result}")
        impressum_result = None
        enrichment_path.append("impressum_error")
    elif impressum_result:
        enrichment_path.append("impressum_ai_extracted")
        exec_names = [e.name for e in (impressum_result.executives or []) if e.name]
        logger.info(f"Impressum: {len(exec_names)} executives ({', '.join(exec_names[:3])}), "
                    f"phones={len(impressum_result.phones)}, "
                    f"address={'yes' if impressum_result.address else 'no'}")

    if isinstance(team_result, Exception):
        logger.warning(f"Team discovery failed: {team_result}")
        team_result = TeamDiscoveryResult(contacts=[], source_urls=[], success=False)
        enrichment_path.append("team_discovery_error")
    elif team_result and team_result.success:
        enrichment_path.append(f"team_discovery_{len(team_result.contacts)}_contacts")
        if team_result.fallback_used:
            enrichment_path.append("team_fallback_linkedin")

    # Process Impressum data
    if impressum_result:
        # Company phone from Impressum
        if impressum_result.phones:
            for phone_data in impressum_result.phones:
                if isinstance(phone_data, dict):
                    number = phone_data.get("number", "")
                    if number and not company_info.phone:
                        company_info.phone = number
                        enrichment_path.append(f"company_phone:impressum->{number[:8]}...")
                        logger.info(f"Company phone from Impressum: {number[:8]}...")
                        break
            if not company_info.phone:
                logger.info(f"Impressum had {len(impressum_result.phones)} phone entries but none usable")
        else:
            logger.info("Impressum scraped successfully but no phone numbers found on page")

        # Company address from Impressum
        if impressum_result.address:
            company_info.address = impressum_result.address
            if not company_info.location:
                company_info.location = impressum_result.address
            enrichment_path.append("impressum_address_found")

        # Collect emails
        for email_data in impressum_result.emails:
            if isinstance(email_data, dict):
                addr = email_data.get("address", "")
                if addr:
                    collected_emails.append(addr)

    # ========== PHASE 3: KANDIDATEN SAMMELN ==========

    # Initialize clients early (needed for DM fallback + Phase 5)
    linkedin_client = LinkedInSearchClient()
    apify_client = get_apify_linkedin_client()

    logger.info("Collecting and validating candidates...")
    all_candidates: List[Dict[str, Any]] = []

    # Priority 1: Contact from job URL (beste Quelle)
    if job_contact and job_contact.name:
        all_candidates.append({
            "name": job_contact.name,
            "email": job_contact.email,
            "title": job_contact.title,
            "phone": job_contact.phone,
            "source": "job_url",
            "priority": 100
        })
        if job_contact.email:
            collected_emails.append(job_contact.email)
        logger.info(f"Job URL contact: {job_contact.name}")

    # Priority 2: Contact from LLM parsing
    if parsed.contact_name:
        # Don't add if already have from job URL
        if not any(_normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(parsed.contact_name) for c in all_candidates):
            all_candidates.append({
                "name": parsed.contact_name,
                "email": parsed.contact_email,
                "phone": parsed.contact_phone,  # Phone from input!
                "source": "llm_parse",
                "priority": 90
            })
            if parsed.contact_email:
                collected_emails.append(parsed.contact_email)
            if parsed.contact_phone:
                logger.info(f"LLM parsed contact: {parsed.contact_name} (with phone: {parsed.contact_phone})")
                enrichment_path.append("input_phone_found")
            else:
                logger.info(f"LLM parsed contact: {parsed.contact_name}")

    # Priority 3: Team page contacts
    if team_result and team_result.contacts:
        # Use correct source based on whether fallback was used
        team_source = "linkedin_fallback" if team_result.fallback_used else "team_page"
        team_priority = 50 if team_result.fallback_used else 70  # Lower priority for fallback

        for contact in team_result.contacts:
            if contact.name and not any(_normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(contact.name) for c in all_candidates):
                all_candidates.append({
                    "name": contact.name,
                    "email": contact.email,
                    "title": contact.title,
                    "source": team_source,
                    "priority": team_priority
                })
                if contact.email:
                    collected_emails.append(contact.email)

        logger.info(f"Team contacts: {len(team_result.contacts)} (source: {team_source})")

    # Priority 4: Executives from Impressum
    if impressum_result and impressum_result.executives:
        for exec_contact in impressum_result.executives:
            if exec_contact.name and not any(_normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(exec_contact.name) for c in all_candidates):
                all_candidates.append({
                    "name": exec_contact.name,
                    "title": exec_contact.title,
                    "source": "impressum",
                    "priority": 50
                })

        logger.info(f"Impressum executives: {len(impressum_result.executives)}")

    # Priority 5: Google Decision-Maker Fallback (when no candidates found)
    # Uses Google to search LinkedIn for HR/Exec at this company
    # ALL results MUST be verified via Apify later (untrusted source)
    if not all_candidates and parsed.company_name:
        logger.info("No candidates found - trying Google decision-maker search as fallback...")
        try:
            dm_candidates = await linkedin_client.find_multiple_decision_makers(
                company=parsed.company_name,
                domain=company_info.domain,
                job_category=payload.category,
                max_candidates=3
            )
            for dm in dm_candidates:
                if dm.get("name") and not any(_normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(dm["name"]) for c in all_candidates):
                    all_candidates.append({
                        "name": dm["name"],
                        "title": dm.get("title"),
                        "linkedin_url": dm.get("linkedin_url"),
                        "source": "linkedin_fallback",  # Untrusted - MUST verify via Apify
                        "priority": 40
                    })
            if dm_candidates:
                enrichment_path.append(f"dm_fallback_{len(dm_candidates)}_found")
                logger.info(f"Decision-maker fallback found {len(dm_candidates)} candidates")
            else:
                enrichment_path.append("dm_fallback_empty")
        except Exception as e:
            logger.warning(f"Decision-maker fallback failed: {e}")
            enrichment_path.append("dm_fallback_error")

    enrichment_path.append(f"total_{len(all_candidates)}_raw_candidates")

    # ========== PHASE 4: KI-VALIDIERUNG & RANKING ==========

    validated_candidates: List[CandidateValidation] = []

    if all_candidates:
        logger.info(f"Validating {len(all_candidates)} candidates with AI...")

        validated_candidates = await validate_and_rank_candidates(
            candidates=all_candidates,
            company_name=parsed.company_name,
            company_domain=company_info.domain,
            job_category=payload.category
        )
        # Note: Tracking happens inside validate_and_rank_candidates (uses Sonnet)

        enrichment_path.append(f"validated_{len(validated_candidates)}_candidates")
        logger.info(f"Validated candidates: {len(validated_candidates)}")

        # Log validation results
        for vc in validated_candidates[:3]:
            logger.info(f"  - {vc.name} (score: {vc.relevance_score}): {vc.validation_notes}")

    # DM fallback: if all candidates failed validation but we had raw candidates
    if not validated_candidates and all_candidates and parsed.company_name:
        logger.info("All candidates failed validation - trying DM fallback...")
        try:
            dm_candidates = await linkedin_client.find_multiple_decision_makers(
                company=parsed.company_name,
                domain=company_info.domain,
                job_category=payload.category,
                max_candidates=3
            )
            if dm_candidates:
                new_candidates = [
                    {"name": dm["name"], "title": dm.get("title"),
                     "linkedin_url": dm.get("linkedin_url"),
                     "source": "linkedin_fallback", "priority": 40}
                    for dm in dm_candidates if dm.get("name")
                ]
                validated_candidates = await validate_and_rank_candidates(
                    candidates=new_candidates,
                    company_name=parsed.company_name,
                    company_domain=company_info.domain,
                    job_category=payload.category
                )
                enrichment_path.append(f"dm_fallback_post_validation_{len(validated_candidates)}_found")
                logger.info(f"DM fallback after validation found {len(validated_candidates)} candidates")
                # Add to all_candidates for later lookup
                all_candidates.extend(new_candidates)
        except Exception as e:
            logger.warning(f"DM fallback after validation failed: {e}")
            enrichment_path.append("dm_fallback_post_validation_error")

    # Take top 3 candidates for verification (try up to 3 if earlier ones fail)
    MAX_CANDIDATES_TO_TRY = 3
    top_candidates = validated_candidates[:MAX_CANDIDATES_TO_TRY]

    # ========== PHASE 5: LINKEDIN SEARCH + EMPLOYMENT VERIFICATION ==========
    #
    # LOGIC:
    # 1. Try up to 3 candidates in order of relevance
    # 2. For each candidate:
    #    - Search LinkedIn URL if not present
    #    - Verify with Apify (is person currently employed there?)
    # 3. Decision:
    #    - Trusted sources (job_url, llm_parse, team_page, impressum):
    #      Keep candidate even WITHOUT LinkedIn (person exists on website)
    #    - Untrusted sources (linkedin_fallback):
    #      ONLY keep if LinkedIn was verified → otherwise try next candidate!
    # 4. Collect up to 2 verified candidates for phone enrichment attempts
    #    (BetterContact/FullEnrich are free on no-result, so 2nd try costs nothing extra)
    MAX_VERIFIED_CANDIDATES = 2  # FullEnrich is free on no-result, so 2nd try costs nothing extra

    TRUSTED_SOURCES = {"job_url", "llm_parse", "team_page", "impressum"}

    verified_candidates: List[CandidateValidation] = []

    for candidate_idx, candidate in enumerate(top_candidates):
        # Stop if we have enough verified candidates
        if len(verified_candidates) >= MAX_VERIFIED_CANDIDATES:
            logger.info(f"Have {MAX_VERIFIED_CANDIDATES} verified candidates, skipping remaining {len(top_candidates) - candidate_idx}")
            break

        logger.info(f"Trying candidate {candidate_idx + 1}/{len(top_candidates)}: {candidate.name}")
        candidate_data = next(
            (c for c in all_candidates if _normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(candidate.name or "")),
            {}
        )

        candidate_source = candidate_data.get("source", "unknown")
        is_trusted = candidate_source in TRUSTED_SOURCES
        linkedin_url = candidate_data.get("linkedin_url")
        linkedin_verified = False

        # Step 1: Search LinkedIn if not present
        if not linkedin_url:
            logger.info(f"Searching LinkedIn for: {candidate.name}")
            found_url = await linkedin_client.find_linkedin_profile(
                name=candidate.name,
                company=parsed.company_name,
                domain=company_info.domain
            )
            if found_url:
                linkedin_url = found_url
                enrichment_path.append(f"linkedin_found_{_safe_first_name(candidate.name)}")
                logger.info(f"Found LinkedIn: {found_url}")

        # Step 1b: AI Name-Matching - intelligently check if LinkedIn matches person
        if linkedin_url:
            try:
                linkedin_slug = linkedin_url.split("/in/")[-1].rstrip("/")
                name_match = await ai_match_linkedin_to_name(
                    linkedin_slug=linkedin_slug,
                    person_name=candidate.name,
                    company_name=parsed.company_name
                )
                # AI decides intelligently - only discard on clear mismatch
                if not name_match.get("matches"):
                    logger.warning(f"AI: LinkedIn doesn't match: {linkedin_url} vs {candidate.name} - {name_match.get('reason', 'no reason')}")
                    enrichment_path.append(f"ai_name_mismatch_{_safe_first_name(candidate.name)}")
                    linkedin_url = None  # Discard non-matching LinkedIn
                else:
                    logger.info(f"AI: LinkedIn name match confirmed: {linkedin_slug} -> {candidate.name}")
                    enrichment_path.append(f"ai_name_match_{_safe_first_name(candidate.name)}")
            except Exception as e:
                logger.warning(f"AI name matching failed: {e} - continuing with LinkedIn URL")
                # On error, continue with LinkedIn URL (will be verified by Apify anyway)

        # Step 2: Verify LinkedIn URL with Apify (for ALL candidates with LinkedIn)
        if linkedin_url:
            logger.info(f"Verifying LinkedIn for: {candidate.name} at {parsed.company_name}")

            try:
                verification = await apify_client.verify_employment(
                    linkedin_url=linkedin_url,
                    expected_company=parsed.company_name
                )
                track_apify(success=True)
                enrichment_path.append(f"apify_verify_{_safe_first_name(candidate.name)}")
            except Exception as e:
                logger.warning(f"Apify verification failed for {candidate.name}: {e}")
                track_apify(success=False)
                # Treat failed verification as not verified - continue without LinkedIn
                linkedin_url = None
                candidate_data["linkedin_url"] = None
                candidate_data["linkedin_verified"] = False
                candidate_data["verification_note"] = f"Apify-Fehler: {str(e)}"
                enrichment_path.append(f"apify_error_{_safe_first_name(candidate.name)}")
                # Skip to decision step (is_trusted check below)
                verification = None

            if verification is None:
                pass  # Skip verification logic, already handled above
            elif verification.is_currently_employed:
                # LinkedIn verified: Name matches AND currently employed
                linkedin_verified = True
                candidate_data["linkedin_url"] = linkedin_url
                candidate_data["linkedin_verified"] = True
                candidate_data["verification_note"] = verification.verification_note
                enrichment_path.append(f"linkedin_verified_{_safe_first_name(candidate.name)}")
                logger.info(f"LINKEDIN VERIFIED: {candidate.name} - {verification.verification_note}")
            else:
                # LinkedIn NOT verified: Wrong person or not employed there
                # DISCARD the LinkedIn URL - don't use it!
                linkedin_verified = False
                candidate_data["linkedin_url"] = None  # Remove invalid LinkedIn
                candidate_data["linkedin_verified"] = False
                candidate_data["verification_note"] = f"LinkedIn verworfen: {verification.verification_note}"
                enrichment_path.append(f"linkedin_discarded_{_safe_first_name(candidate.name)}")
                logger.warning(f"LINKEDIN DISCARDED: {candidate.name} - {verification.verification_note}")

        # Step 3: Decide if candidate should be kept
        if is_trusted:
            # Trusted source: Keep candidate even without verified LinkedIn
            # (Person exists on job posting/website - we know they work there)
            if candidate_source in {"job_url", "llm_parse"}:
                source_note = "Aus aktueller Stellenanzeige"
            else:
                source_note = f"Auf Firmenwebsite ({candidate_source})"

            if linkedin_verified:
                note = f"{source_note} + LinkedIn verifiziert"
            else:
                note = f"{source_note} (ohne verifiziertes LinkedIn)"

            candidate_data["verified_current"] = True
            candidate_data["verification_note"] = note
            verified_candidates.append(candidate)
            enrichment_path.append(f"trusted_{_safe_first_name(candidate.name)}")
            logger.info(f"TRUSTED SOURCE: {candidate.name} - {note}")

        else:
            # Untrusted source (linkedin_fallback): ONLY keep if LinkedIn verified
            if linkedin_verified:
                candidate_data["verified_current"] = True
                verified_candidates.append(candidate)
                enrichment_path.append(f"fallback_verified_{_safe_first_name(candidate.name)}")
                logger.info(f"FALLBACK VERIFIED: {candidate.name} - LinkedIn bestätigt")
            else:
                # No verified LinkedIn = no proof they work there = try next candidate!
                candidate_data["verified_current"] = False
                candidate_data["verification_note"] = "LinkedIn-Fallback ohne Verifizierung - übersprungen"
                enrichment_path.append(f"fallback_skipped_{_safe_first_name(candidate.name)}")
                remaining = len(top_candidates) - candidate_idx - 1
                if remaining > 0:
                    logger.warning(f"FALLBACK SKIPPED: {candidate.name} - arbeitet nicht mehr dort, versuche nächsten Kandidat ({remaining} übrig)")
                else:
                    logger.warning(f"FALLBACK SKIPPED: {candidate.name} - arbeitet nicht mehr dort, keine weiteren Kandidaten")

    # Use verified candidates for phone enrichment
    if verified_candidates:
        top_candidates = verified_candidates
        logger.info(f"Using {len(verified_candidates)} verified candidates for phone enrichment")
    else:
        logger.warning("No verified candidates - no phone enrichment will be attempted")
        top_candidates = []  # Don't use unverified candidates for paid APIs

    # ========== PHASE 6: PHONE ENRICHMENT + COMPANY RESEARCH (parallel) ==========

    # Start company research in background (different APIs, no conflict)
    async def _do_company_research():
        """Run company research + LinkedIn search in parallel with phone enrichment."""
        intel = None
        try:
            researcher = CompanyResearcher()
            intel_result = await researcher.research(
                company_name=parsed.company_name,
                domain=company_info.domain,
                job_description=payload.description,
                job_title=payload.title
            )
            if intel_result and intel_result.summary:
                intel = CompanyIntel(
                    summary=intel_result.summary,
                    description=intel_result.description,
                    industry=intel_result.industry,
                    employee_count=intel_result.employee_count,
                    founded=intel_result.founded,
                    headquarters=intel_result.headquarters,
                    products_services=intel_result.products_services,
                    hiring_signals=intel_result.hiring_signals,
                    website_url=intel_result.website_url
                )
        except Exception as e:
            logger.warning(f"Company research failed: {e}")

        # Also find company LinkedIn
        company_li = None
        if not company_info.linkedin_url and parsed.company_name:
            try:
                company_li = await _google_find_company_linkedin(
                    parsed.company_name,
                    company_info.domain
                )
            except Exception as e:
                logger.warning(f"Company LinkedIn search failed: {e}")

        return intel, company_li

    # Launch company research in background
    company_research_task = asyncio.create_task(_do_company_research())
    if _bg_tasks is not None:
        _bg_tasks.append(company_research_task)

    phone_result: Optional[PhoneResult] = None
    decision_maker: Optional[DecisionMaker] = None

    # First: Check if any candidate already has a phone from input
    for candidate in top_candidates:
        candidate_data = next(
            (c for c in all_candidates if _normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(candidate.name or "")),
            {}
        )
        input_phone = candidate_data.get("phone")

        if input_phone and _is_valid_dach_phone(input_phone):
            logger.info(f"Using phone from input: {input_phone}")
            phone_result = PhoneResult(
                number=input_phone,
                type=PhoneType.UNKNOWN,
                source=PhoneSource.COMPANY_MAIN,  # From job posting
                context_note="Direkt aus Stellenanzeige - wahrscheinlich geschäftlich"
            )
            names = candidate.name.split() if candidate.name else []
            # Only include LinkedIn URL if verified - unverified URLs are worthless
            verified_linkedin = candidate_data.get("linkedin_url") if candidate_data.get("linkedin_verified") else None
            decision_maker = DecisionMaker(
                name=candidate.name,
                first_name=names[0] if names else "",
                last_name=" ".join(names[1:]) if len(names) > 1 else "",
                title=candidate_data.get("title"),
                linkedin_url=verified_linkedin,
                email=candidate.email or candidate_data.get("email"),
                verified_current=candidate_data.get("verified_current", False),
                verification_note=candidate_data.get("verification_note")
            )
            enrichment_path.append("phone_from_input")
            break

    # If no phone from input, try paid APIs
    if not phone_result and not skip_paid_apis and top_candidates:
        logger.info("Starting phone enrichment via APIs...")

        for idx, candidate in enumerate(top_candidates):
            candidate_data = next(
                (c for c in all_candidates if _normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(candidate.name or "")),
                {}
            )

            names = (candidate.name or "").split()
            first_name = names[0] if names else ""
            last_name = " ".join(names[1:]) if len(names) > 1 else ""

            linkedin_url = candidate_data.get("linkedin_url")  # Only set if verified!

            if linkedin_url:
                logger.info(f"Trying enrichment for candidate {idx+1}: {candidate.name} (with verified LinkedIn)")
            else:
                logger.info(f"Trying enrichment for candidate {idx+1}: {candidate.name} (no LinkedIn - FullEnrich only)")

            # BetterContact deaktiviert - nur FullEnrich + Kaspr für Phone
            # phone_result, bc_emails = await _try_bettercontact(...)

            # Try FullEnrich (free on no-result)
            if not phone_result:
                phone_result, fe_emails = await _try_fullenrich(
                    first_name=first_name,
                    last_name=last_name,
                    company_name=parsed.company_name,
                    domain=company_info.domain,
                    linkedin_url=linkedin_url,
                    enrichment_path=enrichment_path
                )
                collected_emails.extend(fe_emails)

            # Kaspr deactivated (no credits remaining)
            # if not phone_result and linkedin_url and candidate_data.get("linkedin_verified") and idx == 0:
            #     phone_result, kaspr_emails = await _try_kaspr(...)

            if phone_result:
                # Found phone - create decision maker with verification status
                # Only include LinkedIn URL if verified - unverified URLs are worthless
                verified_linkedin = linkedin_url if candidate_data.get("linkedin_verified") else None
                decision_maker = DecisionMaker(
                    name=candidate.name,
                    first_name=first_name,
                    last_name=last_name,
                    title=candidate_data.get("title"),
                    linkedin_url=verified_linkedin,
                    email=candidate.email or candidate_data.get("email"),
                    verified_current=candidate_data.get("verified_current", False),
                    verification_note=candidate_data.get("verification_note")
                )
                enrichment_path.append(f"phone_found_candidate_{idx+1}")
                logger.info(f"Phone found via {phone_result.source.value}")
                break

    # Fallback: Use best candidate even without phone
    if not decision_maker and top_candidates:
        best = top_candidates[0]
        candidate_data = next(
            (c for c in all_candidates if _normalize_name_for_dedup(c.get("name", "")) == _normalize_name_for_dedup(best.name or "")),
            {}
        )

        names = best.name.split() if best.name else []
        # Only include LinkedIn URL if verified - unverified URLs are worthless
        verified_linkedin = candidate_data.get("linkedin_url") if candidate_data.get("linkedin_verified") else None
        decision_maker = DecisionMaker(
            name=best.name,
            first_name=names[0] if names else "",
            last_name=" ".join(names[1:]) if len(names) > 1 else "",
            title=candidate_data.get("title"),
            linkedin_url=verified_linkedin,
            email=best.email or candidate_data.get("email"),
            verified_current=candidate_data.get("verified_current", False),
            verification_note=candidate_data.get("verification_note")
        )
        enrichment_path.append("using_best_candidate_no_phone")
        logger.info(f"Using best candidate without phone: {best.name}")

    # ========== COLLECT COMPANY RESEARCH (started in parallel above) ==========

    logger.info("Collecting company research results...")
    company_intel: Optional[CompanyIntel] = None

    try:
        research_result, company_linkedin = await company_research_task
        if research_result:
            company_intel = research_result
            enrichment_path.append("company_research")
            logger.info(f"Company research: {len(research_result.summary)} chars, industry={research_result.industry}")

            # Transfer data
            if company_intel.industry and not company_info.industry:
                company_info.industry = company_intel.industry
            if company_intel.employee_count and not company_info.employee_count:
                company_info.employee_count = company_intel.employee_count

        if company_linkedin and not company_info.linkedin_url:
            company_info.linkedin_url = company_linkedin
            enrichment_path.append("company_linkedin_found")
    except asyncio.CancelledError:
        logger.info("Company research cancelled (pipeline timeout)")
    except Exception as e:
        logger.warning(f"Company research task failed: {e}")
    finally:
        # Ensure background task is cleaned up if still running
        if not company_research_task.done():
            company_research_task.cancel()

    # ========== FINALISIERUNG ==========

    # Deduplicate and clean emails
    unique_emails = list(set(
        e.lower().strip() for e in collected_emails
        if e and '@' in e and not any(x in e.lower() for x in ['.png', '.jpg', '.gif'])
    ))

    # Update decision maker email - ONLY assign email that actually belongs to this person
    if decision_maker and not decision_maker.email and unique_emails and company_info.domain:
        company_domain = (company_info.domain or "").lower().strip().replace('www.', '')

        # Generic email prefixes to exclude
        generic_prefixes = (
            'kontakt@', 'info@', 'contact@', 'bewerbung@', 'jobs@',
            'hinweise@', 'office@', 'mail@', 'service@', 'support@',
            'karriere@', 'personal@', 'hr@', 'team@', 'hello@', 'hallo@'
        )

        personal_emails = [
            e for e in unique_emails
            if (
                not e.startswith(generic_prefixes)
                and e.split('@')[1].lower() == company_domain  # Must match company domain!
            )
        ]

        if personal_emails:
            # Use AI to match email to decision maker (handles umlauts, abbreviations, etc.)
            matching_email = None

            for email in personal_emails:
                # Quick heuristic check first (save API calls for obvious matches)
                email_prefix = email.split('@')[0].lower()
                dm_first = (decision_maker.first_name or "").lower()
                dm_last = (decision_maker.last_name or "").lower()

                # Obvious match: email prefix contains first or last name
                if (dm_first and len(dm_first) >= 3 and dm_first in email_prefix) or \
                   (dm_last and len(dm_last) >= 3 and dm_last in email_prefix):
                    matching_email = email
                    logger.info(f"Email matches name (heuristic): {email} -> {decision_maker.name}")
                    break

            # If no heuristic match, try AI matching for edge cases (umlauts, etc.)
            if not matching_email and personal_emails:
                for email in personal_emails[:3]:  # Max 3 AI calls
                    try:
                        match_result = await ai_match_email_to_person(
                            email=email,
                            person_name=decision_maker.name,
                            company_domain=company_domain
                        )
                        # Note: Tracking happens inside ai_match_email_to_person (uses Haiku)
                        if match_result.get("matches") and match_result.get("confidence") in ["high", "medium"]:
                            matching_email = email
                            logger.info(f"Email matches name (AI): {email} -> {decision_maker.name} ({match_result.get('reason')})")
                            break
                    except Exception as e:
                        logger.warning(f"AI email matching failed: {e}")

            if matching_email:
                decision_maker.email = matching_email
                enrichment_path.append("email_assigned_to_dm")
                logger.info(f"Assigned matching email to decision maker: {matching_email}")
            else:
                # No matching email found - don't assign random email from another person!
                logger.info(f"No email matching '{decision_maker.name}' found - not assigning unrelated email")

    # Determine success and phone status
    success = (
        phone_result is not None or
        company_info.phone is not None or
        len(unique_emails) > 0
    )

    if phone_result:
        phone_status = PhoneStatus.FOUND_MOBILE if phone_result.type == PhoneType.MOBILE else PhoneStatus.FOUND_LANDLINE
    elif skip_paid_apis:
        phone_status = PhoneStatus.SKIPPED_PAID_API
    elif not decision_maker:
        phone_status = PhoneStatus.NO_DECISION_MAKER
    elif not any(p.startswith("linkedin_found_") or p.startswith("linkedin_verified_") for p in enrichment_path):
        phone_status = PhoneStatus.NO_LINKEDIN
    else:
        phone_status = PhoneStatus.API_NO_RESULT

    # Collect warnings (e.g., API key fallback used)
    warnings = get_last_parse_warnings()

    result = EnrichmentResult(
        success=success,
        company=company_info,
        company_intel=company_intel,
        decision_maker=decision_maker,
        phone=phone_result,
        phone_status=phone_status,
        emails=unique_emails,
        enrichment_path=enrichment_path,
        warnings=warnings,
        operational_alerts=operational_alerts,
        job_id=payload.id,
        job_title=payload.title
    )

    # Log cost summary for this enrichment run
    log_cost_summary()

    # Track pipeline result for persistent quality stats
    try:
        track_pipeline_result(result)
    except Exception as e:
        logger.warning(f"Failed to track pipeline stats: {e}")

    logger.info(f"=== Enrichment complete: success={success}, path={' -> '.join(enrichment_path[:10])}... ===")
    return result


async def enrich_lead_test_mode(payload: WebhookPayload) -> EnrichmentResult:
    """Test mode - only uses LLM parsing and free services."""
    return await enrich_lead(payload, skip_paid_apis=True)


# ========== HELPER FUNCTIONS ==========


async def _scrape_job_url_with_ai(
    url: Optional[str],
    company_name: str,
    job_title: Optional[str]
) -> Optional[ExtractedContact]:
    """Scrape job URL and extract contact with AI."""
    if not url:
        return None

    try:
        scraper = JobUrlScraper(timeout=10)
        # Get HTML content
        scraped = await scraper.scrape_contact(url)

        # If traditional scraping found something, return it
        if scraped and scraped.name:
            return ExtractedContact(
                name=scraped.name,
                email=scraped.email,
                phone=scraped.phone,
                title=scraped.title,
                source="job_url"
            )

        # Otherwise try AI extraction on the raw text
        # (This would require modifying JobUrlScraper to expose raw text)
        return None

    except Exception as e:
        logger.warning(f"Job URL scraping failed: {e}")
        return None


async def _scrape_impressum_with_ai(
    domain: Optional[str],
    company_name: str
):
    """Scrape Impressum and extract data with AI."""
    if not domain:
        return None

    try:
        scraper = ImpressumScraper()
        result = await scraper.scrape(
            company_name=company_name,
            domain=domain
        )

        if not result:
            return None

        from clients.ai_extractor import ExtractedImpressum, ExtractedContact, extract_impressum_data

        # Use AI extraction on raw text for executives (proper names + titles)
        ai_result = None
        if result.raw_text and len(result.raw_text.strip()) > 50:
            try:
                ai_result = await extract_impressum_data(
                    page_text=result.raw_text,
                    company_name=company_name
                )
                if ai_result and ai_result.executives:
                    logger.info(f"AI extracted {len(ai_result.executives)} executives from Impressum")
            except Exception as e:
                logger.warning(f"AI Impressum extraction failed, using regex fallback: {e}")

        # Use AI executives if available, otherwise empty (no more fake names from emails)
        executives = ai_result.executives if (ai_result and ai_result.executives) else []

        # Use regex-extracted phones/emails (reliable) + AI address/company_name if available
        return ExtractedImpressum(
            executives=executives,
            phones=[{"number": p.number, "type": "zentrale"} for p in result.phones],
            emails=[{"address": e, "type": "allgemein"} for e in result.emails],
            address=ai_result.address if ai_result and ai_result.address else result.address,
            company_name=ai_result.company_name if ai_result and ai_result.company_name else company_name
        )

    except Exception as e:
        logger.warning(f"Impressum scraping failed: {e}")
        return None


async def _try_bettercontact(
    first_name: str,
    last_name: str,
    company_name: str,
    domain: Optional[str],
    linkedin_url: Optional[str],
    enrichment_path: List[str]
) -> tuple[Optional[PhoneResult], List[str]]:
    """Try BetterContact for phone number (waterfall enrichment)."""
    emails = []

    if not first_name or not last_name:
        return None, emails

    try:
        client = get_bettercontact_client()
        result = await client.enrich(
            first_name=first_name,
            last_name=last_name,
            company_name=company_name,
            domain=domain,
            linkedin_url=linkedin_url
        )

        # Track BetterContact API call
        found_phone = bool(result and result.phones)
        found_email = bool(result and result.emails)
        track_enrichment("BetterContact", success=bool(result), found_phone=found_phone, found_email=found_email)

        if result:
            enrichment_path.append("bettercontact")
            emails.extend(result.emails)

            if result.phones:
                # Filter for DACH phones
                valid_phones = [p for p in result.phones if _is_valid_dach_phone(p.number)]

                if valid_phones:
                    # Prefer mobile
                    mobile_phones = [p for p in valid_phones if p.type == PhoneType.MOBILE]
                    best_phone = mobile_phones[0] if mobile_phones else valid_phones[0]

                    # Add context note - BetterContact uses B2B databases = likely business
                    if best_phone.type == PhoneType.MOBILE:
                        best_phone.context_note = "Aus B2B Datenbank - wahrscheinlich geschäftlich"
                    else:
                        best_phone.context_note = "Aus B2B Datenbank - geschäftliche Festnetznummer"

                    enrichment_path.append("bettercontact_phone_found")
                    track_phone_attempt(
                        service="bettercontact",
                        phones_returned=result.phones,
                        dach_valid_phone=best_phone,
                        phone_type=best_phone.type.value
                    )
                    return best_phone, emails
                else:
                    enrichment_path.append("bettercontact_filtered_non_dach")
                    track_phone_attempt(
                        service="bettercontact",
                        phones_returned=result.phones,
                        dach_valid_phone=None,
                        phone_type=None
                    )
            else:
                track_phone_attempt(
                    service="bettercontact",
                    phones_returned=[],
                    dach_valid_phone=None,
                    phone_type=None
                )

    except Exception as e:
        logger.warning(f"BetterContact failed: {e}")
        track_enrichment("BetterContact", success=False, found_phone=False, found_email=False)
        track_phone_attempt(service="bettercontact", phones_returned=[], dach_valid_phone=None, phone_type=None)

    return None, emails


async def _try_fullenrich(
    first_name: str,
    last_name: str,
    company_name: str,
    domain: Optional[str],
    linkedin_url: Optional[str],
    enrichment_path: List[str]
) -> tuple[Optional[PhoneResult], List[str]]:
    """Try FullEnrich for phone number."""
    emails = []

    if not first_name or not last_name:
        return None, emails

    try:
        client = get_fullenrich_client()
        result = await client.enrich(
            first_name=first_name,
            last_name=last_name,
            company_name=company_name,
            domain=domain,
            linkedin_url=linkedin_url
        )

        # Track FullEnrich API call
        found_phone = bool(result and result.phones)
        found_email = bool(result and result.emails)
        track_enrichment("FullEnrich", success=bool(result), found_phone=found_phone, found_email=found_email)

        if result:
            enrichment_path.append("fullenrich")
            emails.extend(result.emails)

            if result.phones:
                logger.info(f"FullEnrich returned {len(result.phones)} phones: {[p.number[:8]+'...' for p in result.phones]}")
                # Filter for DACH phones
                valid_phones = [p for p in result.phones if _is_valid_dach_phone(p.number)]
                filtered = [p for p in result.phones if not _is_valid_dach_phone(p.number)]
                for p in filtered:
                    logger.info(f"FullEnrich phone filtered (non-DACH): {p.number[:8]}...")

                if valid_phones:
                    # Prefer mobile
                    mobile_phones = [p for p in valid_phones if p.type == PhoneType.MOBILE]
                    best_phone = mobile_phones[0] if mobile_phones else valid_phones[0]

                    # Add context note - FullEnrich uses B2B databases = likely business
                    if best_phone.type == PhoneType.MOBILE:
                        best_phone.context_note = "Aus B2B Datenbank - wahrscheinlich geschäftlich"
                    else:
                        best_phone.context_note = "Aus B2B Datenbank - geschäftliche Festnetznummer"

                    enrichment_path.append("fullenrich_phone_found")
                    track_phone_attempt(
                        service="fullenrich",
                        phones_returned=result.phones,
                        dach_valid_phone=best_phone,
                        phone_type=best_phone.type.value
                    )
                    return best_phone, emails
                else:
                    enrichment_path.append("fullenrich_filtered_non_dach")
                    track_phone_attempt(
                        service="fullenrich",
                        phones_returned=result.phones,
                        dach_valid_phone=None,
                        phone_type=None
                    )
            else:
                track_phone_attempt(
                    service="fullenrich",
                    phones_returned=[],
                    dach_valid_phone=None,
                    phone_type=None
                )

    except Exception as e:
        logger.warning(f"FullEnrich failed: {e}")
        track_enrichment("FullEnrich", success=False, found_phone=False, found_email=False)
        track_phone_attempt(service="fullenrich", phones_returned=[], dach_valid_phone=None, phone_type=None)

    return None, emails


async def _try_kaspr(
    linkedin_url: str,
    name: str,
    enrichment_path: List[str]
) -> tuple[Optional[PhoneResult], List[str]]:
    """Try Kaspr for phone number (requires LinkedIn URL)."""
    emails = []

    try:
        client = get_kaspr_client()
        result = await client.enrich_by_linkedin(
            linkedin_url=linkedin_url,
            name=name
        )

        # Track Kaspr API call
        found_phone = bool(result and result.phones)
        found_email = bool(result and result.emails)
        track_enrichment("Kaspr", success=bool(result), found_phone=found_phone, found_email=found_email)

        if result:
            enrichment_path.append("kaspr")
            emails.extend(result.emails)

            if result.phones:
                logger.info(f"Kaspr returned {len(result.phones)} phones: {[p.number[:8]+'...' for p in result.phones]}")
                valid_phones = [p for p in result.phones if _is_valid_dach_phone(p.number)]
                filtered = [p for p in result.phones if not _is_valid_dach_phone(p.number)]
                for p in filtered:
                    logger.info(f"Kaspr phone filtered (non-DACH): {p.number[:8]}...")

                if valid_phones:
                    mobile_phones = [p for p in valid_phones if p.type == PhoneType.MOBILE]
                    best_phone = mobile_phones[0] if mobile_phones else valid_phones[0]

                    # Kaspr gets data from LinkedIn - likely private number
                    if best_phone.type == PhoneType.MOBILE:
                        best_phone.context_note = "Von LinkedIn Profil - möglicherweise private Nummer"
                    else:
                        best_phone.context_note = "Von LinkedIn Profil - Festnetz"

                    enrichment_path.append("kaspr_phone_found")
                    track_phone_attempt(
                        service="kaspr",
                        phones_returned=result.phones,
                        dach_valid_phone=best_phone,
                        phone_type=best_phone.type.value
                    )
                    return best_phone, emails
                else:
                    enrichment_path.append("kaspr_filtered_non_dach")
                    track_phone_attempt(
                        service="kaspr",
                        phones_returned=result.phones,
                        dach_valid_phone=None,
                        phone_type=None
                    )
            else:
                track_phone_attempt(
                    service="kaspr",
                    phones_returned=[],
                    dach_valid_phone=None,
                    phone_type=None
                )

    except Exception as e:
        logger.warning(f"Kaspr failed: {e}")
        track_enrichment("Kaspr", success=False, found_phone=False, found_email=False)
        track_phone_attempt(service="kaspr", phones_returned=[], dach_valid_phone=None, phone_type=None)

    return None, emails


def _extract_main_domain(domain: str) -> Optional[str]:
    """
    Extract main domain from subdomain.

    Examples:
        professional.dkms.org -> dkms.org (or dkms.de for DACH)
        careers.company.com -> company.com
        www.example.de -> example.de
        portal.jobs.company.de -> company.de

    For .org/.com subdomains of German companies,
    tries to return .de variant as it's more likely to be the main site.
    """
    if not domain:
        return None

    # Clean and lowercase
    domain = domain.lower().strip()

    # Remove www. prefix
    if domain.startswith('www.'):
        domain = domain[4:]

    parts = domain.split('.')

    # Need at least 3 parts for a subdomain (e.g., sub.company.de)
    if len(parts) < 3:
        return domain  # Already a main domain

    # Extract TLD (last part) and check for country-code TLDs
    tld = parts[-1]

    # Handle compound TLDs like .co.uk, .com.br
    if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac', 'gov'):
        # This is like example.co.uk - the main domain is 3 parts
        main_domain = '.'.join(parts[-3:])
    else:
        # Standard TLD - main domain is last 2 parts
        main_domain = '.'.join(parts[-2:])

    # Safety check: don't normalize if the resulting main domain is a
    # large shared/government domain (e.g., bayern.de, bund.de, tum.de)
    # These are too generic — the subdomain IS the actual organization
    shared_domains = {
        'bayern.de', 'bund.de', 'nrw.de', 'sachsen.de', 'hessen.de',
        'hamburg.de', 'berlin.de', 'bremen.de', 'brandenburg.de',
        'thueringen.de', 'niedersachsen.de', 'saarland.de',
        'baden-wuerttemberg.de', 'bwl.de', 'sachsen-anhalt.de',
        'schleswig-holstein.de', 'mecklenburg-vorpommern.de',
        'rlp.de',  # Rheinland-Pfalz
        'gv.at', 'admin.ch',  # Austrian/Swiss government
    }
    if main_domain in shared_domains:
        logger.info(f"Subdomain normalization skipped: {domain} is under shared domain {main_domain}")
        return domain

    # Only normalize known subdomain prefixes (careers, www, jobs, etc.)
    # For deep subdomains (4+ parts like a.b.c.de), keep as-is unless prefix is known
    known_subdomain_prefixes = {
        'www', 'careers', 'career', 'jobs', 'job', 'karriere',
        'recruiting', 'bewerbung', 'stellenangebote', 'apply',
        'hire', 'talent', 'shop', 'store', 'portal', 'app',
        'de', 'en', 'fr', 'it',  # language subdomains
        'professional', 'corporate', 'info', 'web',
    }
    subdomain_prefix = parts[0]
    if len(parts) > 3 and subdomain_prefix not in known_subdomain_prefixes:
        # Deep subdomain with unknown prefix — likely an organizational subdomain, keep it
        logger.info(f"Subdomain normalization skipped: {domain} has deep structure (prefix '{subdomain_prefix}' unknown)")
        return domain

    logger.info(f"Subdomain detected: {domain} -> using main domain {main_domain}")
    return main_domain


# Known job portals - we can't extract company domain from these
JOB_PORTAL_DOMAINS = {
    # German
    'indeed.com', 'indeed.de', 'de.indeed.com',
    'stepstone.de', 'stepstone.at', 'stepstone.ch',
    'monster.de', 'monster.at', 'monster.ch',
    'xing.com', 'linkedin.com',
    'jobware.de', 'stellenanzeigen.de', 'jobs.de',
    'karriere.at', 'jobs.ch', 'jobscout24.de',
    'kimeta.de', 'yourfirm.de', 'glassdoor.de', 'glassdoor.com',
    'kununu.de', 'meinestadt.de', 'hokify.de', 'hokify.at',
    'gigajob.de', 'arbeitsagentur.de', 'jobboerse.de',
    'regio-jobanzeiger.de', 'jobbörse.de', 'jobbörse-stellenangebote.de',
    'experteer.de', 'adzuna.de', 'neuvoo.de', 'jooble.de',
    'stellenwerk.de', 'greenjobs.de', 'azubiyo.de',
    'ausbildung.de', 'praktikum.de', 'campusjäger.de',
    # International / ATS
    'workday.com', 'greenhouse.io', 'lever.co', 'recruitee.com',
    'personio.de', 'softgarden.de', 'recruitingapp.com',
    'icims.com', 'taleo.net', 'successfactors.com',
    'smartrecruiters.com', 'jobvite.com', 'workable.com',
    'join.com', 'onlyfy.com', 'breezy.hr', 'bamboohr.com', 'ashbyhq.com',
    'dvinci.de', 'coveto.de', 'd.vinci.de',
}


def _extract_domain_from_job_url(job_url: Optional[str], company_name: str) -> Optional[str]:
    """
    Extract company domain from job posting URL if it's on the company's website.

    This is the MOST RELIABLE source because:
    - If a job is posted on groeber.de/karriere, the domain IS groeber.de
    - No guessing, no API calls needed

    Returns None if URL is from a job portal (Indeed, Stepstone, etc.)
    """
    if not job_url:
        return None

    try:
        parsed = urlparse(job_url)
        domain = parsed.netloc.lower().replace('www.', '')

        if not domain:
            return None

        # Check if it's a job portal (exact domain or subdomain match)
        if domain in JOB_PORTAL_DOMAINS or any(domain.endswith('.' + portal) for portal in JOB_PORTAL_DOMAINS):
            logger.debug(f"Job URL is from job portal: {domain}")
            return None

        # Additional check: common job portal patterns (subdomain = jobs.company.de)
        if any(pattern in domain for pattern in [
            'jobs.', 'karriere.', 'career.', 'recruiting.', 'bewerbung.',
            'stellenangebote.', 'apply.', 'hire.', 'talent.'
        ]):
            # Could be jobs.company.de - extract main domain
            parts = domain.split('.')
            if len(parts) >= 2:
                main = '.'.join(parts[-2:])  # company.de
                logger.info(f"Extracted main domain from job subdomain: {domain} -> {main}")
                return main

        logger.info(f"✓ Job URL domain extracted: {domain} (from {job_url[:50]}...)")
        return domain

    except Exception as e:
        logger.warning(f"Failed to extract domain from job URL: {e}")
        return None


async def _heuristic_find_domain(company_name: str, job_context: str = "") -> Optional[tuple]:
    """
    Find company domain using smart heuristics - COMPLETELY FREE!

    Strategy:
    1. Extract meaningful words from company name
    2. Generate possible domain variations
    3. Check if domains exist via HEAD requests (free!)
    4. Validate with AI (with job context for smarter validation)

    Returns: (domain, website_url) tuple or None
    Example: "Gröber Holzbau GmbH" → tries groeber.de, groeber.com, groeber-holzbau.de, etc.
    """
    settings = get_settings()

    # Step 1: Normalize company name and extract words
    name_lower = company_name.lower().strip()

    # Remove legal suffixes (longest first to avoid partial matches)
    legal_suffixes = [
        ' gmbh & co. kgaa', ' gmbh & co. kg', ' gmbh & co kg', ' gmbh & co.kg',
        ' gmbh & co. ohg', ' partg mbb', ' partg',
        ' gmbh', ' ggmbh', ' ag', ' kgaa', ' kg', ' ohg', ' mbh',
        ' ug (haftungsbeschränkt)', ' ug', ' gbr', ' eg', ' e.v.', ' e.v',
        ' co.', ' & co', ' inc', ' ltd', ' se', ' sa',
    ]
    for suffix in legal_suffixes:
        name_lower = name_lower.replace(suffix, '')

    name_lower = name_lower.strip()

    # Convert umlauts for domain names
    def to_domain_chars(text: str) -> str:
        """Convert German umlauts to domain-safe characters."""
        replacements = [
            ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
            ('Ä', 'ae'), ('Ö', 'oe'), ('Ü', 'ue')
        ]
        for umlaut, replacement in replacements:
            text = text.replace(umlaut, replacement)
        # Remove any remaining non-domain characters
        text = re.sub(r'[^a-z0-9\-]', '', text)
        return text

    # Get individual words
    words = [w.strip() for w in name_lower.split() if len(w.strip()) >= 2]
    words_normalized = [to_domain_chars(w) for w in words]

    # Primary word (usually company name)
    if not words_normalized:
        logger.warning(f"No usable words in company name: {company_name}")
        return None

    primary = words_normalized[0]

    # Step 2: Generate domain candidates in priority order
    domain_candidates = []

    # TLDs to try (prioritized for DACH region + modern TLDs)
    tlds = ['.de', '.com', '.at', '.ch', '.io', '.eu', '.net', '.org',
            '.tech', '.digital', '.app', '.dev', '.gmbh', '.online']

    # Pattern 1: Primary word only (most common) - e.g., groeber.de
    for tld in tlds:
        domain_candidates.append(f"{primary}{tld}")

    # Pattern 2: All words joined - e.g., groeberholzbau.de
    if len(words_normalized) > 1:
        joined = ''.join(words_normalized)
        for tld in tlds[:5]:  # Main TLDs including .io
            domain_candidates.append(f"{joined}{tld}")

    # Pattern 3: Words with hyphen - e.g., groeber-holzbau.de
    if len(words_normalized) > 1:
        hyphenated = '-'.join(words_normalized)
        for tld in tlds[:5]:
            domain_candidates.append(f"{hyphenated}{tld}")

    # Pattern 4: Reversed hyphen - e.g., holzbau-groeber.de
    if len(words_normalized) == 2:
        reversed_hyphen = f"{words_normalized[1]}-{words_normalized[0]}"
        for tld in tlds[:3]:
            domain_candidates.append(f"{reversed_hyphen}{tld}")

    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for d in domain_candidates:
        if d not in seen:
            seen.add(d)
            unique_candidates.append(d)

    logger.info(f"Heuristic domain candidates for '{company_name}': {unique_candidates[:10]}")

    # Step 3: Check which domains exist AND fetch content to detect parked domains
    async def check_domain_exists(domain: str) -> tuple[str, bool, str, str]:
        """Check if domain exists, has a website, and is NOT a parked domain.
        Returns (domain, exists, page_snippet, scheme) - snippet is first ~5000 chars of HTML."""
        try:
            async with httpx.AsyncClient(
                timeout=7.0,
                follow_redirects=True,
                verify=False,  # Some sites have cert issues
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            ) as client:
                # Use GET instead of HEAD to get page content for parking detection
                for scheme in ["https", "http"]:
                    try:
                        response = await client.get(
                            f"{scheme}://{domain}",
                            follow_redirects=True,
                            headers={"Accept": "text/html"}
                        )
                        if response.status_code < 400:
                            content = response.text[:5000] if response.text else ""
                            if _is_parked_domain(content, domain):
                                logger.info(f"Parked domain detected: {domain} - skipping")
                                return (domain, False, "", "")
                            return (domain, True, content[:3000], scheme)
                    except Exception as e:
                        logger.debug(f"Domain {domain} HTTP check failed: {e}")
                        continue
                return (domain, False, "", "")
        except Exception as e:
            logger.debug(f"Domain {domain} HTTP check failed: {e}")
            return (domain, False, "", "")

    # Check first 15 candidates in parallel
    tasks = [check_domain_exists(d) for d in unique_candidates[:15]]
    results = await asyncio.gather(*tasks)

    # Filter to existing, non-parked domains (keep content snippets + scheme for AI validation)
    existing_domains = [(domain, snippet, scheme) for domain, exists, snippet, scheme in results if exists]

    if not existing_domains:
        logger.info(f"No heuristic domains exist for '{company_name}'")
        return None

    logger.info(f"Existing domains for '{company_name}': {[d for d, _, _ in existing_domains]}")

    # Step 4: Validate with AI (with page content + job context for smarter validation)
    for domain, snippet, scheme in existing_domains:
        is_valid, ai_reason = await _ai_validate_domain(domain, company_name, page_content=snippet, job_context=job_context)
        if is_valid:
            website_url = f"{scheme}://{domain}"
            logger.info(f"✓ Heuristic found valid domain: {domain} for '{company_name}'")
            return (domain, website_url)

    logger.warning(f"No heuristic domain passed AI validation for '{company_name}'")
    return None


async def _serper_places_lookup(
    company_name: str,
    location: str = "",
) -> Optional[dict]:
    """
    Serper Places API — Google Maps/Business data (Firmenkarte).
    Returns dict with keys: domain, website, phone, address, title (or None).
    Costs 1 Serper credit ($0.001).
    """
    settings = get_settings()
    if not settings.serper_api_key:
        return None

    query = f"{company_name} {location}".strip() if location else company_name

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://google.serper.dev/places",
                headers={
                    "X-API-KEY": settings.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "gl": "de", "hl": "de", "location": "Germany"},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning(f"Serper Places API call failed: {e}")
        return None

    places = data.get("places", [])
    if not places:
        logger.info(f"Serper Places: no results for '{company_name}'")
        return None

    # Take first result — Google Maps usually returns the best match first
    place = places[0]
    place_title = place.get("title", "")
    place_website = place.get("website") or ""
    place_phone = place.get("phoneNumber") or ""
    place_address = place.get("address") or ""

    logger.info(f"Serper Places for '{company_name}': title={place_title}, website={place_website}, phone={place_phone}")

    # Simple title overlap check — at least one significant word must match
    name_words = {w.lower() for w in company_name.split() if len(w) > 2}
    title_words = {w.lower() for w in place_title.split() if len(w) > 2}
    if name_words and not name_words & title_words:
        logger.info(f"Serper Places: title mismatch — '{place_title}' vs '{company_name}', skipping")
        return None

    result = {"title": place_title}

    # Extract domain from website URL
    if place_website:
        try:
            parsed = urlparse(place_website if place_website.startswith("http") else f"https://{place_website}")
            domain = parsed.netloc.lower().replace("www.", "")
            if domain and not any(skip in domain for skip in SKIP_DOMAINS):
                result["domain"] = domain
                result["website"] = f"https://{domain}"
        except Exception:
            pass

    # Normalize phone (local "089 xxx" → "+49 89 xxx")
    if place_phone:
        normalized = normalize_phone_number(place_phone, default_region="DE")
        if normalized and _is_valid_dach_phone(normalized):
            result["phone"] = normalized
        elif _is_valid_dach_phone(place_phone):
            # Fallback: keep original if it passes DACH validation
            result["phone"] = place_phone

    if place_address:
        result["address"] = place_address

    return result if ("domain" in result or "phone" in result) else None


async def _serper_find_domain(
    company_name: str,
    job_context: str = "",
    location: str = "",
) -> Optional[tuple]:
    """
    Find company domain via Serper.dev Google SERP API.
    Runs Places + Search in parallel, then AI picks the best result.

    Costs 2 Serper credits ($0.002) — 1 for search, 1 for places.

    Returns: (domain, website_url, phone, address) tuple or None
    """
    settings = get_settings()
    if not settings.serper_api_key:
        return None

    query = f"{company_name} {location}".strip() if location else company_name
    logger.info(f"Serper domain search for: {query}")

    # Step 1: Run Places + Search in parallel
    async def _serper_search():
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": settings.serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json={"q": query, "gl": "de", "hl": "de", "num": 10},
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"Serper Search API call failed: {e}")
            return None

    search_data, places_data = await asyncio.gather(
        _serper_search(),
        _serper_places_lookup(company_name, location),
    )

    if not search_data and not places_data:
        logger.warning(f"Serper: both Search and Places failed for '{company_name}'")
        return None

    # Extract Knowledge Graph data from search results (legacy fallback)
    kg = (search_data or {}).get("knowledgeGraph", {})
    kg_phone = kg.get("phoneNumber") or kg.get("telephone") or None
    kg_address = kg.get("address") or None
    kg_website = kg.get("website") or None

    if kg_phone or kg_address or kg_website:
        logger.info(f"Serper KG for '{company_name}': website={kg_website}, phone={kg_phone}, address={kg_address}")

    # Step 2: Build AI input with ALL data sources
    ai_sections = []

    # Places data
    if places_data:
        places_line = f"  Title: {places_data.get('title', 'N/A')}"
        if places_data.get('domain'):
            places_line += f" | Website: {places_data['domain']}"
        if places_data.get('phone'):
            places_line += f" | Tel: {places_data['phone']}"
        if places_data.get('address'):
            places_line += f" | Adresse: {places_data['address']}"
        ai_sections.append(f"Google Places (Firmenkarte):\n{places_line}")

    # Knowledge Graph data
    if kg_website or kg_phone:
        kg_line = f"  Website: {kg_website or 'N/A'}"
        if kg_phone:
            kg_line += f" | Tel: {kg_phone}"
        if kg_address:
            kg_line += f" | Adresse: {kg_address}"
        ai_sections.append(f"Google Knowledge Graph:\n{kg_line}")

    # Organic search results
    organic_lines = []
    organic_results = (search_data or {}).get("organic", [])
    for i, result in enumerate(organic_results[:10], 1):
        link = result.get("link", "")
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        if not link:
            continue
        parsed_url = urlparse(link)
        domain = parsed_url.netloc.lower().replace("www.", "")
        line = f"  {i}. \"{title}\" | {domain}"
        if snippet:
            line += f" | Snippet: \"{snippet[:150]}\""
        organic_lines.append(line)

    if organic_lines:
        ai_sections.append("Google Suche Ergebnisse:\n" + "\n".join(organic_lines))

    if not ai_sections:
        logger.info(f"Serper: no data to analyze for '{company_name}'")
        return None

    # Build AI prompt
    job_hint = f" (Jobtitel/Kontext: {job_context[:200]})" if job_context else ""
    all_data = "\n\n".join(ai_sections)

    ai_prompt = f"""Firma: "{company_name}"{job_hint}

{all_data}

Welches Ergebnis ist die RICHTIGE Firmenwebseite für "{company_name}"?
Beachte:
- Die Domain muss zur EINSTELLENDEN FIRMA gehören, NICHT zu Jobportalen oder Personalvermittlern
- Google Places (Firmenkarte) ist sehr zuverlässig — bevorzuge diese wenn der Firmenname passt
- Snippets können die richtige Domain oder Telefonnummer enthalten (z.B. "Web: firma.de", "Telefon: +49...")
- Extrahiere Telefonnummer und Adresse wenn verfügbar (aus Places, KG oder Snippets)
- Bei ähnlichen Firmennamen: Achte auf EXAKTE Übereinstimmung (z.B. "089 Apartments" ≠ "089 Immobilienmanagement")

Antworte NUR mit JSON:
{{"domain": "firma.de", "phone": "+49...", "address": "Straße, Stadt", "source": "places/kg/organic/snippet", "confidence": "high/medium/low", "reason": "kurze Begründung"}}

Wenn KEINE passende Domain gefunden: {{"domain": null, "reason": "..."}}"""

    try:
        llm_client = get_llm_client()
        response = await llm_client.call(
            prompt=ai_prompt,
            tier="balanced",  # Haiku 4.5 — fast + accurate for structured selection
            max_tokens=200,
        )

        import json
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        ai_result = json.loads(content)
        chosen_domain = ai_result.get("domain")
        ai_phone = ai_result.get("phone") or None
        ai_address = ai_result.get("address") or None
        ai_source = ai_result.get("source", "unknown")
        ai_confidence = ai_result.get("confidence", "unknown")
        ai_reason = ai_result.get("reason", "")

        logger.info(f"Serper AI chose domain={chosen_domain}, phone={ai_phone}, source={ai_source}, confidence={ai_confidence}, reason={ai_reason}")

        if not chosen_domain:
            logger.info(f"Serper AI: no suitable domain for '{company_name}' — {ai_reason}")
            return None

    except Exception as e:
        logger.warning(f"Serper AI selection failed: {e} — falling back to places/KG domain")
        # Fallback: use Places or KG domain without AI
        chosen_domain = (places_data or {}).get("domain")
        if not chosen_domain and kg_website:
            try:
                parsed_kg = urlparse(kg_website if kg_website.startswith("http") else f"https://{kg_website}")
                chosen_domain = parsed_kg.netloc.lower().replace("www.", "")
            except Exception:
                pass
        ai_phone = (places_data or {}).get("phone") or kg_phone
        ai_address = (places_data or {}).get("address") or kg_address
        ai_source = "fallback"

        if not chosen_domain:
            return None

    # Step 3: HTTP-check + parked domain check on AI-chosen domain
    try:
        async with httpx.AsyncClient(
            timeout=7.0, follow_redirects=True, verify=False,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        ) as http_client:
            domain_reachable = False
            final_url = f"https://{chosen_domain}"
            for scheme in ["https", "http"]:
                try:
                    resp = await http_client.get(
                        f"{scheme}://{chosen_domain}",
                        follow_redirects=True,
                        headers={"Accept": "text/html"}
                    )
                    if resp.status_code < 400:
                        content = resp.text[:5000] if resp.text else ""
                        if _is_parked_domain(content, chosen_domain):
                            logger.info(f"Serper: AI-chosen domain {chosen_domain} is parked — rejecting")
                            return None
                        final_url = f"{scheme}://{chosen_domain}"
                        domain_reachable = True
                        break
                except Exception:
                    continue

            if not domain_reachable:
                logger.warning(f"Serper: AI-chosen domain {chosen_domain} not reachable — trying organic fallback")

                # Step 3b: Fallback — try top organic domains individually
                fallback_candidates = []
                seen = {chosen_domain}
                for result in organic_results:
                    link = result.get("link", "")
                    if not link:
                        continue
                    parsed_url = urlparse(link)
                    fb_d = parsed_url.netloc.lower().replace("www.", "")
                    if fb_d in seen or not fb_d:
                        continue
                    if any(skip in fb_d for skip in SKIP_DOMAINS):
                        continue
                    seen.add(fb_d)
                    fallback_candidates.append(fb_d)

                fallback_candidates.sort(key=lambda d: -_domain_relevance_score(d, company_name))

                for fb_domain in fallback_candidates[:3]:
                    try:
                        for scheme in ["https", "http"]:
                            try:
                                resp = await http_client.get(
                                    f"{scheme}://{fb_domain}",
                                    follow_redirects=True,
                                    headers={"Accept": "text/html"},
                                )
                                if resp.status_code < 400:
                                    fb_content = resp.text[:5000] if resp.text else ""
                                    if _is_parked_domain(fb_content, fb_domain):
                                        break
                                    ai_valid, _ = await _ai_validate_domain(fb_domain, company_name, fb_content[:3000], job_context)
                                    if ai_valid:
                                        chosen_domain = fb_domain
                                        final_url = f"{scheme}://{fb_domain}"
                                        domain_reachable = True
                                        logger.info(f"✓ Serper fallback domain: {fb_domain}")
                                        break
                            except Exception:
                                continue
                        if domain_reachable:
                            break
                    except Exception:
                        continue

                if not domain_reachable:
                    logger.warning(f"Serper: no reachable domain found for '{company_name}'")
                    return None
    except Exception as e:
        logger.warning(f"Serper: HTTP check for {chosen_domain} failed: {e}")
        return None

    # Step 4: Validate phone if AI extracted one
    final_phone = None
    if ai_phone:
        # Normalize phone number
        normalized = normalize_phone_number(ai_phone, default_region="DE")
        if normalized and _is_valid_dach_phone(normalized):
            final_phone = normalized
        elif _is_valid_dach_phone(ai_phone):
            final_phone = ai_phone

    # Fallback: use Places or KG phone if AI didn't find one
    if not final_phone:
        places_phone = (places_data or {}).get("phone")
        if places_phone and _is_valid_dach_phone(places_phone):
            final_phone = places_phone
        elif kg_phone and _is_valid_dach_phone(kg_phone):
            normalized = normalize_phone_number(kg_phone, default_region="DE")
            final_phone = normalized or kg_phone

    final_address = ai_address or (places_data or {}).get("address") or kg_address

    logger.info(f"✓ Serper found domain: {chosen_domain} (source={ai_source}, phone={'yes' if final_phone else 'no'}, address={'yes' if final_address else 'no'})")
    return (chosen_domain, final_url, final_phone, final_address)


async def _knowledge_graph_find_domain(
    company_name: str,
    job_context: str = "",
) -> Optional[tuple]:
    """
    Find company domain via Google Knowledge Graph API.
    FREE tier: 100,000 queries/day using existing Google API key.

    Returns: (domain, website_url) tuple or None
    """
    settings = get_settings()
    if not settings.google_api_key:
        return None

    logger.info(f"Knowledge Graph search for: {company_name}")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                "https://kgsearch.googleapis.com/v1/entities:search",
                params={
                    "query": company_name,
                    "key": settings.google_api_key,
                    "types": "Organization",
                    "languages": "de",
                    "limit": 5,
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        logger.warning(f"Knowledge Graph API call failed: {e}")
        return None

    for item in data.get("itemListElement", []):
        result_score = item.get("resultScore", 0)
        if result_score < 100:
            continue

        result = item.get("result", {})
        website = result.get("url")
        if not website:
            continue

        # Parse domain from the official website URL
        try:
            parsed_url = urlparse(website)
            domain = parsed_url.netloc.lower().replace("www.", "")
        except Exception:
            continue

        if not domain:
            continue

        # Skip known non-company domains
        if any(skip in domain for skip in SKIP_DOMAINS):
            continue

        logger.info(f"Knowledge Graph candidate: {domain} (score: {result_score})")

        # HTTP reachability + parked domain check (same pattern as Serper/DDG)
        page_content = ""
        try:
            async with httpx.AsyncClient(
                timeout=7.0, follow_redirects=True, verify=False,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            ) as http_client:
                for scheme in ["https", "http"]:
                    try:
                        resp = await http_client.get(
                            f"{scheme}://{domain}",
                            follow_redirects=True,
                            headers={"Accept": "text/html"}
                        )
                        if resp.status_code < 400:
                            content = resp.text[:5000] if resp.text else ""
                            if _is_parked_domain(content, domain):
                                logger.info(f"Knowledge Graph: parked domain skipped: {domain}")
                                page_content = ""
                                break
                            page_content = content[:3000]
                            break
                    except Exception as e:
                        logger.debug(f"KG domain {domain} HTTP check failed: {e}")
                        continue
        except Exception as e:
            logger.debug(f"KG domain {domain} HTTP check error: {e}")

        if not page_content:
            # Domain unreachable or parked - skip
            logger.info(f"Knowledge Graph: domain {domain} unreachable or parked - skipping")
            continue

        # Validate with AI (now with page content)
        ai_valid, ai_reason = await _ai_validate_domain(domain, company_name, page_content=page_content, job_context=job_context)
        if ai_valid:
            website_url = f"https://{domain}"
            logger.info(f"✓ Knowledge Graph found valid domain: {domain} for '{company_name}'")
            return (domain, website_url)

    logger.info(f"Knowledge Graph: no valid domain for '{company_name}'")
    return None


async def _duckduckgo_find_domain(
    company_name: str,
    job_title: str = "",
    job_context: str = "",
    location: str = "",
) -> Optional[tuple]:
    """
    Find company domain via DuckDuckGo search - FREE, no API key needed.

    Uses ddgs library for better result quality than the old HTML endpoint
    scraping approach. Runs sync DDGS in a thread to avoid blocking the event loop.

    Returns: (domain, website_url) tuple or None
    Raises: Exception on search errors (DDG broken/rate-limited)
    """
    from ddgs import DDGS

    query = f"{company_name} {location}".strip() if location else company_name
    logger.info(f"DuckDuckGo domain search for: {query}")

    def _do_search():
        return DDGS(verify=False).text(query, region="de-de", max_results=10)

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(_do_search), timeout=10
        )
    except asyncio.TimeoutError:
        raise RuntimeError("DuckDuckGo search timed out after 10s")
    except Exception as e:
        raise RuntimeError(f"DuckDuckGo search failed: {e}")

    if not results:
        raise RuntimeError("DuckDuckGo returned no results - possibly rate-limited")

    # Extract candidate domains with snippets and actual URLs
    candidate_domains = []  # (domain, snippet, website_url)
    seen_domains = set()

    for result in results:
        href = result.get("href", "")
        snippet = result.get("body", "")
        if not href:
            continue

        parsed_url = urlparse(href)
        domain = parsed_url.netloc.lower().replace("www.", "")

        if not domain or domain in seen_domains:
            continue
        if any(skip in domain for skip in SKIP_DOMAINS):
            continue

        seen_domains.add(domain)
        website_url = f"{parsed_url.scheme}://{domain}"
        candidate_domains.append((domain, snippet, website_url))

    if not candidate_domains:
        logger.info(f"DuckDuckGo: no valid candidate domains for '{company_name}'")
        return None

    # Sort by relevance, take top 5
    scored = [(d, s, u, _domain_relevance_score(d, company_name)) for d, s, u in candidate_domains]
    scored.sort(key=lambda x: -x[3])

    logger.info(f"DuckDuckGo candidates for '{company_name}': {[(d, sc) for d, _, _, sc in scored[:5]]}")

    # HTTP-check + parked-domain-check for top 5 candidates (reuse heuristic pattern)
    async def check_ddg_domain(domain: str, website_url: str) -> tuple[str, str, str, bool]:
        """Check if DDG candidate domain is reachable and not parked.
        Returns (domain, website_url, page_content, is_valid)."""
        try:
            async with httpx.AsyncClient(
                timeout=7.0,
                follow_redirects=True,
                verify=False,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            ) as client:
                for scheme in ["https", "http"]:
                    try:
                        response = await client.get(
                            f"{scheme}://{domain}",
                            follow_redirects=True,
                            headers={"Accept": "text/html"}
                        )
                        if response.status_code < 400:
                            content = response.text[:5000] if response.text else ""
                            if _is_parked_domain(content, domain):
                                logger.info(f"DuckDuckGo: parked domain skipped: {domain}")
                                return (domain, website_url, "", False)
                            return (domain, f"{scheme}://{domain}", content[:3000], True)
                    except Exception as e:
                        logger.debug(f"DuckDuckGo domain {domain} HTTP check failed: {e}")
                        continue
            logger.info(f"DuckDuckGo: domain not reachable: {domain}")
            return (domain, website_url, "", False)
        except Exception:
            logger.info(f"DuckDuckGo: domain check failed: {domain}")
            return (domain, website_url, "", False)

    # Check top 5 candidates in parallel
    top5 = scored[:5]
    check_tasks = [check_ddg_domain(d, u) for d, _, u, _ in top5]
    check_results = await asyncio.gather(*check_tasks)

    # AI-validate only reachable, non-parked domains (with real page content)
    for domain, website_url, page_content, is_valid in check_results:
        if not is_valid:
            continue
        ai_valid, ai_reason = await _ai_validate_domain(
            domain, company_name,
            page_content=page_content,
            job_context=job_context,
        )
        if ai_valid:
            logger.info(f"✓ DuckDuckGo found valid domain: {domain} for '{company_name}'")
            return (domain, website_url)

    logger.info(f"DuckDuckGo: no domain passed AI validation for '{company_name}'")
    return None


def _is_parked_domain(html_content: str, domain: str = "") -> bool:
    """
    Detect if a domain is parked/for-sale based on page content.
    Uses simple string matching - completely FREE, no API calls.
    """
    if not html_content or len(html_content.strip()) < 50:
        return True  # Empty or near-empty page = likely parked

    content_lower = html_content.lower()

    # Known parking/domain-sale services
    parking_services = [
        'sedo.com', 'sedo.de', 'sedoparking',
        'godaddy.com', 'godaddy parking',
        'afternic.com', 'afternic',
        'hugedomains.com', 'hugedomains',
        'dan.com',  # Domain marketplace
        'undeveloped.com',
        'domainmarket.com',
        'porkbun.com/domain',
        'namecheap.com/domains',
        'ionos.de', 'ionos.com',
        'united-domains.de',
        'strato.de/domains',
        'checkdomain.de',
        'domainssaubillig',
        'parkingcrew',
        'bodis.com',  # Parking service
        'domainparking',
        'above.com',
    ]

    for service in parking_services:
        if service in content_lower:
            return True

    # Common parking/for-sale phrases (German + English)
    parking_phrases = [
        'diese domain kaufen',
        'domain kaufen',
        'domain zu verkaufen',
        'domain steht zum verkauf',
        'diese domain ist zu verkaufen',
        'diese webseite ist zu verkaufen',
        'domain is for sale',
        'this domain is for sale',
        'buy this domain',
        'domain for sale',
        'website for sale',
        'is for sale',
        'domain name for sale',
        'this page is parked',
        'domain parked',
        'parked domain',
        'parked by',
        'domain erwerben',
        'domain sichern',
        'sponsored listings',
        'related searches',  # Typical parking page content
        'click here to buy',
        'make an offer',
        'inquire about this domain',
        'domain anfragen',
    ]

    match_count = 0
    for phrase in parking_phrases:
        if phrase in content_lower:
            match_count += 1
            if match_count >= 1:  # Even one strong indicator is enough
                return True

    # Check for very thin content (parked pages have almost no real text)
    # Strip HTML tags for content length check
    import re as _re
    text_only = _re.sub(r'<[^>]+>', ' ', html_content)
    text_only = _re.sub(r'\s+', ' ', text_only).strip()
    if len(text_only) < 100:
        return True

    return False


async def _ai_validate_domain(domain: str, company_name: str, page_content: str = "", job_context: str = "") -> tuple[bool, str]:
    """
    Use AI to check if a domain belongs to the company.
    Includes page content snippet and job context for smarter validation.
    Returns (matches: bool, reason: str) tuple.
    """
    try:
        llm_client = get_llm_client()

        # Build context from page content if available
        content_context = ""
        if page_content:
            # Strip HTML tags for a cleaner snippet
            import re as _re
            text_snippet = _re.sub(r'<[^>]+>', ' ', page_content[:2000])
            text_snippet = _re.sub(r'\s+', ' ', text_snippet).strip()[:500]
            if text_snippet:
                content_context = f'\nInhalt der Webseite (Ausschnitt): "{text_snippet}"'

        # Build job context hint if available
        job_hint = ""
        if job_context:
            job_hint = f'\nKontext der Stellenanzeige: "{job_context[:300]}"'

        prompt = f"""Prüfe ob die Domain "{domain}" zur Firma "{company_name}" gehört.
{content_context}{job_hint}
WICHTIG:
- Die Domain muss zur EINSTELLENDEN FIRMA gehören
- NICHT zu einer Personalvermittlung, Recruiting-Agentur oder Jobportal
- Wenn der Webseiteninhalt vorhanden ist, prüfe ob er zur Firma passt
- Wenn Stellenanzeigen-Kontext vorhanden ist, prüfe ob die Branche/Tätigkeit zur Webseite passt
- Bei gleichem Firmennamen aber ANDERER Branche/Tätigkeit = KEIN Match
- Eine Domain-Kauf-Seite oder leere Seite = KEIN Match

Antworte NUR mit JSON:
{{"matches": true/false, "reason": "kurze Begründung"}}"""

        response = await llm_client.call(
            prompt=prompt,
            tier="fast",  # Fast and cheap for simple validation
            max_tokens=100
        )

        import json
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        result = json.loads(content)
        matches = result.get("matches", False)
        reason = result.get("reason", "")

        if matches:
            logger.info(f"AI validated domain: {domain} -> {company_name} ✓ ({reason})")
        else:
            logger.warning(f"AI rejected domain: {domain} -> {company_name} ✗ ({reason})")

        return (matches, reason)

    except Exception as e:
        logger.error(f"AI domain validation failed: {e} - rejecting domain for safety")
        return (False, f"validation_error: {e}")  # On error, reject - better no domain than wrong domain


async def _google_find_domain(company_name: str, validate_with_ai: bool = True) -> Optional[str]:
    """
    Find company domain via Google Custom Search.

    Args:
        company_name: The company name to search for
        validate_with_ai: If True, validates each found domain with AI

    Returns up to 3 candidate domains and validates them.
    """
    settings = get_settings()

    if not settings.google_api_key or not settings.google_cse_id:
        return None

    async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
        # Better query: just company name, no "official website" which doesn't help for German companies
        query = f'"{company_name}"'
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": settings.google_api_key,
            "cx": settings.google_cse_id,
            "q": query,
            "num": 10  # Get more results to have alternatives
        }

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            track_google("domain_search")  # Track Google API call
            data = response.json()

            # Collect ALL candidate domains first
            candidate_domains = []
            for item in data.get("items", []):
                link = item.get("link", "")
                if link:
                    parsed = urlparse(link)
                    domain = parsed.netloc.replace("www.", "")
                    if not any(skip in domain for skip in SKIP_DOMAINS):
                        if domain not in candidate_domains:
                            candidate_domains.append(domain)

            if not candidate_domains:
                logger.warning(f"No candidate domains found for {company_name}")
                return None

            # Sort by relevance using shared scoring
            candidate_domains_scored = [(d, _domain_relevance_score(d, company_name)) for d in candidate_domains]
            candidate_domains_scored.sort(key=lambda x: -x[1])

            logger.info(f"Google CSE candidates for '{company_name}': {[(d, s) for d, s in candidate_domains_scored[:5]]}")

            # Take top 5 for HTTP check + AI validation
            top5 = [d for d, s in candidate_domains_scored[:5]]

            if validate_with_ai:
                # HTTP-check + parked-domain-check (same pattern as DDG)
                async def _check_cse_domain(domain: str) -> tuple:
                    try:
                        async with httpx.AsyncClient(
                            timeout=7.0, follow_redirects=True, verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                        ) as http_client:
                            for scheme in ["https", "http"]:
                                try:
                                    resp = await http_client.get(
                                        f"{scheme}://{domain}",
                                        follow_redirects=True,
                                        headers={"Accept": "text/html"}
                                    )
                                    if resp.status_code < 400:
                                        content = resp.text[:5000] if resp.text else ""
                                        if _is_parked_domain(content, domain):
                                            return (domain, "", False)
                                        return (domain, content[:3000], True)
                                except Exception as e:
                                    logger.debug(f"CSE domain {domain} HTTP check failed: {e}")
                                    continue
                        return (domain, "", False)
                    except Exception as e:
                        logger.debug(f"CSE domain {domain} HTTP check failed: {e}")
                        return (domain, "", False)

                check_results = await asyncio.gather(*[_check_cse_domain(d) for d in top5])

                for domain, page_content, is_valid in check_results:
                    if not is_valid:
                        continue
                    ai_valid, ai_reason = await _ai_validate_domain(domain, company_name, page_content=page_content)
                    if ai_valid:
                        return domain

                logger.warning(f"No domain passed validation for {company_name} - candidates: {top5}")
                return None
            else:
                return top5[0]

        except Exception as e:
            logger.warning(f"Google domain search failed: {e}")

    return None


async def _google_find_company_linkedin(
    company_name: str,
    domain: Optional[str] = None
) -> Optional[str]:
    """Find company LinkedIn page via Google."""
    settings = get_settings()

    if not settings.google_api_key or not settings.google_cse_id:
        return None

    async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
        if domain:
            query = f'"{company_name}" OR "{domain}" site:linkedin.com/company'
        else:
            query = f'"{company_name}" site:linkedin.com/company'

        params = {
            "key": settings.google_api_key,
            "cx": settings.google_cse_id,
            "q": query,
            "num": 3
        }

        try:
            response = await client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            response.raise_for_status()
            track_google("company_linkedin_search")  # Track Google API call
            data = response.json()

            for item in data.get("items", []):
                link = item.get("link", "")
                if "linkedin.com/company/" in link:
                    return link.split("?")[0]

        except Exception as e:
            logger.warning(f"Google company LinkedIn search failed: {e}")

    return None
