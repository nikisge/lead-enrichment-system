import json
import re
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from anthropic import AsyncAnthropic

from config import get_settings
from models import WebhookPayload, ParsedJobPosting

logger = logging.getLogger(__name__)


@dataclass
class _ParseState:
    """Per-request parse state (thread/task-safe via ContextVar)."""
    used_fallback: bool = False
    warning: Optional[str] = None


_parse_state: ContextVar[_ParseState] = ContextVar('_parse_state')


def _get_parse_state() -> _ParseState:
    """Get or create per-context parse state (lazy init avoids shared mutable default)."""
    try:
        return _parse_state.get()
    except LookupError:
        state = _ParseState()
        _parse_state.set(state)
        return state


def get_last_parse_warnings() -> List[str]:
    """Get warnings from the last parse operation."""
    state = _get_parse_state()
    warnings = []
    if state.used_fallback:
        warnings.append("primary_api_key_failed")
        warnings.append("used_fallback_api_key")
    if state.warning:
        warnings.append(state.warning)
    return warnings

def reset_parse_warnings():
    """Reset warnings for new parse operation."""
    _parse_state.set(_ParseState())


SYSTEM_PROMPT = """Du bist ein Experte für die Analyse von Stellenanzeigen im DACH-Raum.
Extrahiere strukturierte Informationen aus der Stellenanzeige.

WICHTIG für company_domain:
- Extrahiere die Domain NUR wenn eine WEBSITE explizit im Text erwähnt wird (z.B. "www.firma.de", "firma.de")
- IGNORIERE Email-Domains komplett! Email-Adressen sind oft von Personalvermittlungen, nicht vom Unternehmen.
- Beispiel: Bei "Bewerbung an m.jaeger@pletschacher.de" für "Gröber Holzbau GmbH" → company_domain = null (NICHT pletschacher.de!)
- Setze company_domain auf null wenn du dir nicht 100% sicher bist

Regeln:
- Suche nach genannten Ansprechpartnern (oft am Ende: "Ihr Ansprechpartner", "Kontakt", "Bewerbung an")
- Extrahiere E-Mail-Adressen falls vorhanden (für Kontakt, NICHT für Domain!)
- Extrahiere Telefonnummern falls vorhanden (Format: +49, 0049, oder 0xxx)
- Bestimme relevante Titel für Entscheider (HR, Personal, Geschäftsführung)

Antworte NUR mit validem JSON im folgenden Format (keine anderen Texte):
{
    "company_name": "Firmenname",
    "company_domain": "firma.de (NUR aus Website-Erwähnung, NICHT aus Email!) oder null",
    "contact_name": "Vorname Nachname oder null",
    "contact_email": "email@firma.de oder null",
    "contact_phone": "+49 123 456789 oder null",
    "target_titles": ["HR Manager", "Personalleiter"],
    "department": "HR/Personal/IT/etc oder null",
    "location": "Stadt, Land oder null"
}"""


async def parse_job_posting(payload: WebhookPayload) -> ParsedJobPosting:
    """
    Use LLM to extract structured info from job posting.
    Falls back to regex extraction if LLM fails.
    Uses fallback API key if primary key fails.
    """
    reset_parse_warnings()
    settings = get_settings()

    # Try LLM parsing with primary key first, then fallback
    api_keys = []
    if settings.anthropic_api_key:
        api_keys.append(("primary", settings.anthropic_api_key))
    if settings.anthropic_api_key_fallback:
        api_keys.append(("fallback", settings.anthropic_api_key_fallback))

    for key_type, api_key in api_keys:
        try:
            result = await _llm_parse(payload, api_key)
            if key_type == "fallback":
                _get_parse_state().used_fallback = True
                logger.warning("PRIMARY API KEY FAILED - Used fallback API key successfully")
            return result
        except Exception as e:
            error_msg = str(e)
            if "credit balance" in error_msg.lower():
                logger.error(f"API key ({key_type}) has no credits: {e}")
                if key_type == "primary":
                    logger.info("Trying fallback API key...")
                    continue
            elif "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                logger.error(f"API key ({key_type}) is invalid: {e}")
                if key_type == "primary":
                    logger.info("Trying fallback API key...")
                    continue
            else:
                logger.warning(f"LLM parsing failed with {key_type} key: {e}")
                if key_type == "primary":
                    continue

    # All API keys failed - use regex fallback
    logger.warning("All API keys failed, using regex fallback")
    _get_parse_state().warning = "llm_parse_failed_used_regex_fallback"
    return _regex_parse(payload)


async def _llm_parse(payload: WebhookPayload, api_key: str) -> ParsedJobPosting:
    """Parse using Claude Sonnet."""
    client = AsyncAnthropic(api_key=api_key)

    user_content = f"""Stellenanzeige:
Firma: {payload.company}
Titel: {payload.title}
Ort: {payload.location or 'Nicht angegeben'}

Beschreibung:
{payload.description[:6000]}"""

    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_content}
        ]
    )

    content = response.content[0].text

    # Extract JSON from response (Claude might add some text around it)
    # Support nested objects by finding balanced braces
    content = content.strip()

    # Remove markdown code blocks if present
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Try to parse directly first
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Find JSON object with balanced braces
        start_idx = content.find('{')
        if start_idx != -1:
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            content = content[start_idx:end_idx]
        data = json.loads(content)

    # Ensure target_titles has defaults if empty
    if not data.get("target_titles"):
        data["target_titles"] = _get_default_titles(payload.title)

    # Remove any extra fields not in ParsedJobPosting
    allowed_fields = {"company_name", "company_domain", "contact_name", "contact_email",
                      "contact_phone", "target_titles", "department", "location"}
    data = {k: v for k, v in data.items() if k in allowed_fields}

    return ParsedJobPosting(**data)


def _regex_parse(payload: WebhookPayload) -> ParsedJobPosting:
    """
    Fallback regex-based parsing.

    WICHTIG: Dies ist nur der Fallback wenn AI komplett fehlschlägt.
    Wir setzen domain=None und lassen die Pipeline später via Google Search
    die richtige Domain finden. Besser keine Domain als eine falsche!
    """
    description = payload.description
    company_name = payload.company

    # Extract email
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    emails = re.findall(email_pattern, description)
    contact_email = emails[0] if emails else None

    # Extract phone numbers (German formats)
    phone_pattern = r'(?:\+49|0049|0)\s*[\d\s\-/()]{8,20}'
    phones = re.findall(phone_pattern, description)
    contact_phone = None
    if phones:
        # Clean and take first valid phone
        for phone in phones:
            cleaned = re.sub(r'[^\d+]', '', phone)
            if len(cleaned) >= 10:
                contact_phone = phone.strip()
                break

    # NICHT automatisch Domain extrahieren im Fallback!
    # Das war der Bug: Email-Domain von Personalvermittlung wurde verwendet.
    # Besser: Domain = None setzen, Pipeline macht dann Google Search.
    domain = None
    logger.warning(f"Regex fallback: Setting domain=None for '{company_name}' - Pipeline will use Google Search")

    # Extract contact name (common patterns)
    contact_name = None
    patterns = [
        r'[Aa]nsprechpartner(?:in)?[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
        r'[Kk]ontakt[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
        r'[Ii]hr[e]?\s+[Aa]nsprechpartner(?:in)?[:\s]+([A-ZÄÖÜ][a-zäöüß]+\s+[A-ZÄÖÜ][a-zäöüß]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, description)
        if match:
            contact_name = match.group(1).strip()
            break

    return ParsedJobPosting(
        company_name=payload.company,
        company_domain=domain,  # None - let Pipeline find via Google
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        target_titles=_get_default_titles(payload.title),
        department=_detect_department(payload.title, payload.category),
        location=payload.location
    )


def _get_default_titles(job_title: str) -> List[str]:
    """Get relevant decision maker titles based on job posting."""
    job_lower = job_title.lower()

    # HR/Personnel related
    if any(x in job_lower for x in ['hr', 'personal', 'recruiting', 'talent']):
        return [
            "HR Manager", "HR-Manager", "Personalleiter", "Personalleiterin",
            "Head of HR", "HR Director", "Leiter Personal",
            "Recruiting Manager", "Head of Recruiting"
        ]

    # IT related
    if any(x in job_lower for x in ['it', 'software', 'developer', 'engineer', 'tech', 'consultant']):
        return [
            "IT-Leiter", "Head of IT", "CTO", "IT Manager",
            "Leiter Softwareentwicklung", "Head of Engineering",
            "HR Manager", "Personalleiter"
        ]

    # Sales related
    if any(x in job_lower for x in ['sales', 'vertrieb', 'account']):
        return [
            "Vertriebsleiter", "Head of Sales", "Sales Director",
            "Leiter Vertrieb", "HR Manager", "Personalleiter"
        ]

    # Default: HR + Management
    return [
        "HR Manager", "Personalleiter", "Personalleiterin",
        "Geschäftsführer", "Geschäftsführerin", "CEO",
        "Head of HR", "Leiter Personal"
    ]


def _detect_department(job_title: str, category: Optional[str]) -> Optional[str]:
    """Detect department from job title or category."""
    text = f"{job_title} {category or ''}".lower()

    if any(x in text for x in ['hr', 'personal', 'recruiting']):
        return "HR"
    if any(x in text for x in ['it', 'software', 'tech', 'developer', 'consultant']):
        return "IT"
    if any(x in text for x in ['sales', 'vertrieb']):
        return "Sales"
    if any(x in text for x in ['marketing']):
        return "Marketing"
    if any(x in text for x in ['finance', 'finanz', 'accounting']):
        return "Finance"

    return None
