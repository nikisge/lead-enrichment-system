"""
AI-based Data Extraction for Lead Enrichment.

Uses LLMs to intelligently extract contact information from:
- Team pages
- Impressum pages
- Job posting pages

Replaces error-prone regex extraction with contextual AI understanding.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from clients.llm_client import get_llm_client, ModelTier
from utils.cost_tracker import track_llm

logger = logging.getLogger(__name__)

# Maximum characters to send to LLM (context protection)
MAX_LLM_INPUT_CHARS = 12000


@dataclass
class ExtractedContact:
    """A contact person extracted from a page."""
    name: str
    title: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    source: str = ""
    confidence: float = 0.8


@dataclass
class ExtractedImpressum:
    """Extracted data from an Impressum page."""
    executives: List[ExtractedContact] = field(default_factory=list)
    phones: List[Dict[str, str]] = field(default_factory=list)  # {number, type}
    emails: List[Dict[str, str]] = field(default_factory=list)  # {address, type}
    address: Optional[str] = None
    company_name: Optional[str] = None


def truncate_text(text: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> str:
    """
    Truncate text intelligently for LLM input.
    Keeps beginning and end (Impressum often at end).
    """
    if len(text) <= max_chars:
        return text

    # Keep 60% from start, 40% from end
    start_chars = int(max_chars * 0.6)
    end_chars = max_chars - start_chars - 50  # Leave room for truncation marker

    return text[:start_chars] + "\n\n[... Inhalt gekürzt ...]\n\n" + text[-end_chars:]


async def extract_contacts_from_page(
    page_text: str,
    company_name: str,
    page_type: str = "team"
) -> List[ExtractedContact]:
    """
    Extract contact persons from any page using AI.

    Args:
        page_text: Raw text content from the page
        company_name: Company name for context
        page_type: Type of page (team, impressum, job_posting, about)

    Returns:
        List of extracted contacts
    """
    if not page_text or len(page_text.strip()) < 50:
        logger.info(f"Page text too short for extraction: {len(page_text)} chars")
        return []

    # Truncate if needed
    text = truncate_text(page_text)

    llm = get_llm_client()

    prompt = f"""Analysiere diesen {page_type}-Text von "{company_name}" und extrahiere alle echten Mitarbeiter/Ansprechpartner.

WICHTIG - Extrahiere NUR:
- Echte Personennamen (Vor- und Nachname)
- KEINE Überschriften, Menüpunkte oder Platzhalter
- KEINE generischen Texte wie "Unser Team" oder "Kontaktieren Sie uns"

Für jeden gefundenen Mitarbeiter gib zurück:
- name: Vollständiger Name (Vor- und Nachname)
- title: Position/Jobtitel falls vorhanden (sonst null)
- email: E-Mail-Adresse falls vorhanden (sonst null)
- phone: Telefonnummer falls vorhanden (sonst null)

Text:
{text}

Antworte als JSON-Array:
[{{"name": "Max Müller", "title": "Geschäftsführer", "email": "m.mueller@firma.de", "phone": null}}]

Falls keine echten Personen gefunden werden: []"""

    result = await llm.call_json(prompt, tier=ModelTier.BALANCED)
    track_llm("contact_extract", tier="sonnet")  # Contact extraction uses Sonnet

    if not result or not isinstance(result, list):
        logger.info(f"No contacts extracted from {page_type} page")
        return []

    contacts = []
    for item in result:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "").strip()
        if not name or len(name) < 3:
            continue

        # Basic validation: name should have at least 2 words
        if len(name.split()) < 2:
            continue

        contacts.append(ExtractedContact(
            name=name,
            title=item.get("title"),
            email=item.get("email"),
            phone=item.get("phone"),
            source=page_type
        ))

    logger.info(f"Extracted {len(contacts)} contacts from {page_type} page")
    return contacts


async def extract_impressum_data(
    page_text: str,
    company_name: str
) -> ExtractedImpressum:
    """
    Extract structured data from an Impressum page.

    Extracts:
    - Geschäftsführer (important contacts!)
    - Phone numbers with type (Zentrale, Mobil, Fax)
    - Email addresses with type (Allgemein, Persönlich)
    - Address

    Args:
        page_text: Raw text from Impressum page
        company_name: Company name for context

    Returns:
        ExtractedImpressum with all found data
    """
    if not page_text or len(page_text.strip()) < 50:
        return ExtractedImpressum()

    text = truncate_text(page_text, max_chars=8000)

    llm = get_llm_client()

    prompt = f"""Extrahiere aus diesem Impressum-Text von "{company_name}" alle relevanten Informationen.

WICHTIG:
- Geschäftsführer/Inhaber sind wichtige Kontaktpersonen!
- Unterscheide zwischen persönlichen und allgemeinen Kontaktdaten

Extrahiere:
1. executives: Geschäftsführer, Inhaber, Vorstände mit Name und Titel
2. phones: Alle Telefonnummern mit Typ (zentrale/mobil/fax/direkt)
3. emails: Alle E-Mails mit Typ (allgemein/persönlich/support)
4. address: Vollständige Adresse
5. company_name: Offizieller Firmenname aus dem Impressum

Text:
{text}

Antworte als JSON:
{{
    "executives": [{{"name": "Max Müller", "title": "Geschäftsführer"}}],
    "phones": [{{"number": "+49 89 123456", "type": "zentrale"}}],
    "emails": [{{"address": "info@firma.de", "type": "allgemein"}}],
    "address": "Musterstraße 1, 80333 München",
    "company_name": "Firma GmbH"
}}"""

    result = await llm.call_json(prompt, tier=ModelTier.FAST)
    track_llm("impressum_extract", tier="haiku")  # Impressum extraction uses Haiku

    if not result or not isinstance(result, dict):
        logger.info("No Impressum data extracted")
        return ExtractedImpressum()

    # Parse executives
    executives = []
    for exec_data in result.get("executives", []):
        if isinstance(exec_data, dict) and exec_data.get("name"):
            name = exec_data["name"].strip()
            if len(name.split()) >= 2:  # At least first + last name
                executives.append(ExtractedContact(
                    name=name,
                    title=exec_data.get("title"),
                    source="impressum"
                ))

    return ExtractedImpressum(
        executives=executives,
        phones=result.get("phones", []),
        emails=result.get("emails", []),
        address=result.get("address"),
        company_name=result.get("company_name")
    )


async def extract_job_posting_contact(
    page_text: str,
    company_name: str,
    job_title: Optional[str] = None
) -> Optional[ExtractedContact]:
    """
    Extract the contact person from a job posting page.

    Looks for patterns like:
    - "Ihr Ansprechpartner: Name"
    - "Kontakt: Name"
    - "Bewerbung an: Name"

    Args:
        page_text: Raw text from job posting page
        company_name: Company name for context
        job_title: Job title for context

    Returns:
        ExtractedContact if found, None otherwise
    """
    if not page_text or len(page_text.strip()) < 100:
        return None

    text = truncate_text(page_text, max_chars=8000)

    llm = get_llm_client()

    job_context = f" für die Stelle '{job_title}'" if job_title else ""

    prompt = f"""Analysiere diese Stellenanzeige von "{company_name}"{job_context}.

Finde den Ansprechpartner/Kontakt für Bewerbungen.

Suche nach Mustern wie:
- "Ihr Ansprechpartner: ..."
- "Kontakt: ..."
- "Bewerbung an: ..."
- "Fragen? Kontaktieren Sie ..."
- "Frau/Herr ..."

WICHTIG:
- Nur ECHTE Personennamen (Vor- und Nachname)
- Keine generischen Texte oder Abteilungsnamen
- Keine Firmennamen

Text:
{text}

Falls ein Ansprechpartner gefunden wurde, antworte als JSON:
{{"name": "Max Müller", "title": "HR Manager", "email": "max.mueller@firma.de", "phone": null}}

Falls KEIN Ansprechpartner gefunden wurde:
{{"name": null}}"""

    result = await llm.call_json(prompt, tier=ModelTier.FAST)
    track_llm("contact_extract", tier="haiku")  # Job contact extraction uses Haiku

    if not result or not isinstance(result, dict):
        return None

    name = result.get("name")
    if not name or len(name.strip()) < 3:
        return None

    name = name.strip()

    # Validate: at least 2 words (first + last name)
    if len(name.split()) < 2:
        logger.info(f"Job contact name too short: {name}")
        return None

    contact = ExtractedContact(
        name=name,
        title=result.get("title"),
        email=result.get("email"),
        phone=result.get("phone"),
        source="job_posting",
        confidence=0.9  # High confidence for job posting contacts
    )

    logger.info(f"Extracted job contact: {contact.name} ({contact.title})")
    return contact


async def extract_contacts_with_priority(
    page_text: str,
    company_name: str,
    job_category: Optional[str] = None
) -> List[ExtractedContact]:
    """
    Extract contacts and prioritize by relevance.

    Priority:
    1. HR / Recruiting (100)
    2. Department head matching job category (80)
    3. Executives / Geschäftsführer (60)
    4. Other named contacts (40)

    Args:
        page_text: Page content
        company_name: Company name
        job_category: Optional job category for relevance scoring

    Returns:
        List of contacts sorted by priority
    """
    contacts = await extract_contacts_from_page(page_text, company_name, "team")

    if not contacts:
        return []

    # Build priority scoring prompt
    llm = get_llm_client()

    contacts_data = [
        {"name": c.name, "title": c.title or "Unbekannt"}
        for c in contacts
    ]

    category_hint = f"\nDie Stelle ist im Bereich: {job_category}" if job_category else ""

    prompt = f"""Bewerte diese Kontakte von "{company_name}" nach Relevanz als Ansprechpartner für Bewerbungen.{category_hint}

Kontakte:
{contacts_data}

Bewertungskriterien:
- HR/Personal/Recruiting: 100 Punkte
- Abteilungsleiter passend zur Stelle: 80 Punkte
- Geschäftsführer/CEO/Inhaber: 60 Punkte
- Sonstige: 40 Punkte

Antworte als JSON-Array, sortiert nach Priorität (höchste zuerst):
[{{"name": "...", "priority": 100}}]"""

    result = await llm.call_json(prompt, tier=ModelTier.FAST)

    if not result or not isinstance(result, list):
        return contacts

    # Create priority map
    priority_map = {}
    for item in result:
        if isinstance(item, dict) and item.get("name"):
            priority_map[item["name"].lower()] = item.get("priority", 50)

    # Sort contacts by priority
    def get_priority(contact: ExtractedContact) -> int:
        return priority_map.get(contact.name.lower(), 50)

    contacts.sort(key=get_priority, reverse=True)

    return contacts


async def ai_match_linkedin_to_name(
    linkedin_slug: str,
    person_name: str,
    company_name: Optional[str] = None
) -> dict:
    """
    Use AI to check if a LinkedIn URL slug matches a person's name.

    Handles edge cases like:
    - Umlauts: "glaetzle" should match "Alexander Glätzle"
    - Abbreviations: "m-mueller" should match "Max Müller"
    - Nicknames: "alex-schmidt" should match "Alexander Schmidt"

    Args:
        linkedin_slug: The slug from LinkedIn URL (e.g., "glaetzle", "max-mueller-123abc")
        person_name: Full name we're looking for (e.g., "Alexander Glätzle")
        company_name: Optional company for additional context

    Returns:
        {"matches": True/False, "confidence": "high"/"medium"/"low", "reason": "..."}
    """
    llm = get_llm_client()

    company_context = f" bei {company_name}" if company_name else ""

    prompt = f"""Prüfe ob dieser LinkedIn-URL-Slug zur gesuchten Person passt.

LinkedIn-Slug: "{linkedin_slug}"
Gesuchte Person: "{person_name}"{company_context}

Beachte:
- Umlaute werden in URLs ersetzt: ü→u, ä→a, ö→o, ß→ss
- Slugs können Zahlen am Ende haben (z.B. "max-mueller-123abc")
- Manchmal nur Nachname im Slug (z.B. "glaetzle" für "Alexander Glätzle")
- Spitznamen möglich (z.B. "alex" für "Alexander")

Antworte als JSON:
{{"matches": true/false, "confidence": "high/medium/low", "reason": "Kurze Begründung"}}"""

    result = await llm.call_json(prompt, tier=ModelTier.FAST)
    track_llm("linkedin_match", tier="haiku")  # LinkedIn matching uses Haiku

    if result and isinstance(result, dict):
        return {
            "matches": result.get("matches", False),
            "confidence": result.get("confidence", "low"),
            "reason": result.get("reason", "")
        }

    return {"matches": False, "confidence": "low", "reason": "KI-Auswertung fehlgeschlagen"}


async def ai_match_email_to_person(
    email: str,
    person_name: str,
    company_domain: Optional[str] = None
) -> dict:
    """
    Use AI to check if an email address belongs to a specific person.

    Handles edge cases like:
    - Umlauts: "mueller@firma.de" should match "Max Müller"
    - Initials: "m.mueller@firma.de" should match "Max Müller"
    - Variations: "maxm@firma.de" should match "Max Müller"

    Args:
        email: Email address to check
        person_name: Person's name to match against
        company_domain: Company domain for validation

    Returns:
        {"matches": True/False, "confidence": "high"/"medium"/"low", "reason": "..."}
    """
    llm = get_llm_client()

    domain_context = f"\nFirmen-Domain: {company_domain}" if company_domain else ""

    prompt = f"""Prüfe ob diese E-Mail-Adresse zu dieser Person gehört.

E-Mail: "{email}"
Person: "{person_name}"{domain_context}

Beachte:
- Umlaute: ü→ue/u, ä→ae/a, ö→oe/o, ß→ss
- Formate: vorname.nachname@, v.nachname@, vorname@, nachname@
- Die E-Mail-Domain sollte zur Firma passen

Beispiele:
- "mueller@firma.de" → "Max Müller" ✓
- "m.mueller@firma.de" → "Max Müller" ✓
- "johannes@planqc.eu" → "Johannes Zeiher" ✓
- "johannes@planqc.eu" → "Alexander Glätzle" ✗ (falscher Vorname!)

Antworte als JSON:
{{"matches": true/false, "confidence": "high/medium/low", "reason": "Kurze Begründung"}}"""

    result = await llm.call_json(prompt, tier=ModelTier.FAST)
    track_llm("email_match", tier="haiku")  # Email matching uses Haiku

    if result and isinstance(result, dict):
        return {
            "matches": result.get("matches", False),
            "confidence": result.get("confidence", "low"),
            "reason": result.get("reason", "")
        }

    return {"matches": False, "confidence": "low", "reason": "KI-Auswertung fehlgeschlagen"}


async def is_valid_person_name(name: str) -> bool:
    """
    Check if a string is a valid person name using AI.

    Returns True for real names, False for:
    - HTML artifacts
    - Menu items
    - Generic text
    - Company names
    """
    if not name or len(name) < 3:
        return False

    # Quick heuristic checks first (save API calls)
    name_lower = name.lower()

    # Obvious non-names
    obvious_invalid = [
        'weitere', 'möglichkeiten', 'helfen', 'navigation', 'menü',
        'kontakt', 'impressum', 'startseite', 'übersicht', 'angebot',
        'unsere', 'unser', 'team', 'mehr erfahren', 'weiterlesen'
    ]

    if any(word in name_lower for word in obvious_invalid):
        return False

    # Must have at least 2 words
    if len(name.split()) < 2:
        return False

    # For borderline cases, use AI
    llm = get_llm_client()

    prompt = f"""Ist "{name}" ein echter deutscher Personenname (Vor- und Nachname)?

Antworte NUR mit:
{{"valid": true}} oder {{"valid": false}}

Ungültig sind:
- Überschriften ("Weitere Möglichkeiten")
- Menüpunkte ("Navigation überspringen")
- Generische Texte
- Firmennamen
- Jobtitel ohne Namen"""

    result = await llm.call_json(prompt, tier=ModelTier.FAST)

    if result and isinstance(result, dict):
        return result.get("valid", False)

    return False
