import json
import re
import logging
from typing import Optional, List
from anthropic import AsyncAnthropic

from config import get_settings
from models import WebhookPayload, ParsedJobPosting

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Du bist ein Experte für die Analyse von Stellenanzeigen im DACH-Raum.
Extrahiere strukturierte Informationen aus der Stellenanzeige.

Wichtig:
- Suche nach genannten Ansprechpartnern (oft am Ende: "Ihr Ansprechpartner", "Kontakt", "Bewerbung an")
- Extrahiere E-Mail-Adressen falls vorhanden
- Extrahiere Telefonnummern falls vorhanden (Format: +49, 0049, oder 0xxx)
- Identifiziere die Firma und versuche die Domain abzuleiten (aus Email oder Website-Erwähnungen)
- Bestimme relevante Titel für Entscheider (HR, Personal, Geschäftsführung)
- Wenn eine Website wie www.firma.de erwähnt wird, nutze "firma.de" als Domain

Antworte NUR mit validem JSON im folgenden Format (keine anderen Texte):
{
    "company_name": "Firmenname",
    "company_domain": "firma.de oder null",
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
    """
    settings = get_settings()

    # Try LLM parsing first
    if settings.anthropic_api_key:
        try:
            return await _llm_parse(payload, settings.anthropic_api_key)
        except Exception as e:
            logger.warning(f"LLM parsing failed, using fallback: {e}")

    # Fallback to regex-based extraction
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

    return ParsedJobPosting(**data)


def _regex_parse(payload: WebhookPayload) -> ParsedJobPosting:
    """Fallback regex-based parsing."""
    description = payload.description

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

    # Extract domain from email, website mention, or company name
    domain = None

    # Try from email first
    if contact_email:
        domain = contact_email.split('@')[1]

    # Try from website mentions (www.xyz.de or xyz.de)
    if not domain:
        website_pattern = r'(?:www\.)?([a-zA-Z0-9-]+\.(?:de|com|at|ch|eu|io|net|org))'
        website_match = re.search(website_pattern, description)
        if website_match:
            domain = website_match.group(1)

    # Fallback: derive from company name
    if not domain:
        company_clean = payload.company.lower()
        company_clean = re.sub(r'\s*(gmbh|ag|kg|ohg|mbh|ug|se|co\.?|&).*$', '', company_clean, flags=re.IGNORECASE)
        company_clean = re.sub(r'[^a-z0-9]', '', company_clean)
        if company_clean:
            domain = f"{company_clean}.de"

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
        company_domain=domain,
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
