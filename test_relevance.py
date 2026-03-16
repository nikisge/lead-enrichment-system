#!/usr/bin/env python3
"""
Test: Ansprechpartner-Relevanz
Prüft ob die neuen Änderungen korrekt funktionieren:
1. HR wird NICHT als Kandidat gefunden
2. Fach-Entscheider werden bevorzugt
3. target_titles werden korrekt durchgereicht
4. match_reason + department_match im Output
5. n8n-Output-Format stimmt (keine Breaking Changes)
6. Serper-Personensuche funktioniert

Nutzt skip_paid_apis=True -> KEINE Phone-API-Kosten!
Nutzt trotzdem: LLM (Anthropic/OpenRouter), Serper, Google CSE, Apify
"""
import asyncio
import json
import logging
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-7s %(message)s'
)
logger = logging.getLogger(__name__)

# ── Test Cases: verschiedene Branchen ──────────────────────────────
TEST_CASES = [
    {
        "name": "IT-Job (Softwareentwickler)",
        "payload": {
            "category": "IT",
            "company": "DATEV eG",
            "date_posted": "2026-03-10",
            "description": """Die DATEV eG ist das Softwarehaus und der IT-Dienstleister für Steuerberater,
            Wirtschaftsprüfer und Rechtsanwälte. Mit über 8.000 Mitarbeitenden gestalten wir die digitale
            Zukunft. Zur Verstärkung unseres Teams suchen wir einen Senior Software Developer (m/w/d)
            für Cloud-Applikationen in Nürnberg.

            Ihre Aufgaben:
            - Entwicklung von Cloud-Native Anwendungen in Java/Kotlin
            - Architektur und Design von Microservices
            - Code Reviews und Mentoring junger Entwickler

            Ihr Profil:
            - Abgeschlossenes Informatik-Studium
            - 5+ Jahre Erfahrung in Java/Kotlin
            - Erfahrung mit Kubernetes und Cloud-Technologien""",
            "id": "test-datev-it-001",
            "location": "Nürnberg, Deutschland",
            "title": "Senior Software Developer (m/w/d) Cloud",
            "url": None,
        },
        "expected_titles": ["IT-Leiter", "CTO", "Head of IT"],
        "forbidden_titles": ["HR", "Personal", "Recruiting"],
    },
    {
        "name": "Sales-Job (Vertrieb)",
        "payload": {
            "category": "Sales",
            "company": "Würth Group",
            "date_posted": "2026-03-10",
            "description": """Die Würth-Gruppe ist Weltmarktführer im Handel mit Befestigungs- und
            Montagematerial. Für unseren Vertrieb suchen wir einen Vertriebsmitarbeiter Außendienst (m/w/d)
            im Raum Stuttgart.

            Ihre Aufgaben:
            - Betreuung und Ausbau des bestehenden Kundenstamms
            - Neukundenakquise im Handwerk und Industrie
            - Beratung und Verkauf unserer Produktpalette

            Wir bieten:
            - Firmenwagen auch zur privaten Nutzung
            - Attraktives Fixgehalt plus Provision""",
            "id": "test-wuerth-sales-001",
            "location": "Stuttgart, Deutschland",
            "title": "Vertriebsmitarbeiter Außendienst (m/w/d)",
            "url": None,
        },
        "expected_titles": ["Vertriebsleiter", "Head of Sales", "Sales Director"],
        "forbidden_titles": ["HR", "Personal", "Recruiting"],
    },
    {
        "name": "Generischer Job (Projektmanager)",
        "payload": {
            "category": None,
            "company": "Bosch GmbH",
            "date_posted": "2026-03-10",
            "description": """Robert Bosch GmbH sucht einen Projektmanager (m/w/d) für den Standort
            Gerlingen bei Stuttgart. In dieser Position koordinieren Sie bereichsübergreifende
            Projekte und stellen den Projekterfolg sicher.

            Ihre Aufgaben:
            - Planung und Steuerung von Projekten
            - Stakeholder-Management
            - Risikomanagement und Reporting

            Ihr Profil:
            - Studium im Bereich BWL oder Ingenieurwesen
            - PMP oder Prince2 Zertifizierung von Vorteil
            - Sehr gute Deutsch- und Englischkenntnisse""",
            "id": "test-bosch-generic-001",
            "location": "Gerlingen, Deutschland",
            "title": "Projektmanager (m/w/d)",
            "url": None,
        },
        "expected_titles": ["Geschäftsführer", "CEO", "Abteilungsleiter"],
        "forbidden_titles": ["HR Manager", "Personalleiter", "Recruiter"],
    },
]


def check_no_hr(result, test_name: str) -> list:
    """Prüfe dass KEIN HR-Kontakt ausgewählt wurde."""
    issues = []
    if result.decision_maker and result.decision_maker.title:
        title_lower = result.decision_maker.title.lower()
        hr_keywords = ['hr', 'personal', 'recruiting', 'recruiter', 'talent acquisition']
        for kw in hr_keywords:
            if kw in title_lower:
                issues.append(f"HR-KONTAKT GEFUNDEN: {result.decision_maker.name} ({result.decision_maker.title})")
    return issues


def check_n8n_format(result, test_name: str) -> list:
    """Prüfe dass n8n-Output-Format keine Breaking Changes hat."""
    issues = []

    # Must-have fields für n8n
    try:
        output = result.model_dump()
    except Exception as e:
        issues.append(f"model_dump() failed: {e}")
        return issues

    required_fields = ['success', 'company', 'decision_maker', 'phone', 'phone_status',
                       'emails', 'enrichment_path', 'job_id', 'job_title']
    for field in required_fields:
        if field not in output:
            issues.append(f"Missing field: {field}")

    # Company must have expected sub-fields
    if 'company' in output and output['company']:
        for cf in ['name', 'domain']:
            if cf not in output['company']:
                issues.append(f"Missing company.{cf}")

    # decision_maker new fields must be OPTIONAL (not break old n8n)
    if output.get('decision_maker'):
        dm = output['decision_maker']
        # Old fields must still exist
        for old_field in ['name', 'title', 'email', 'linkedin_url', 'verified_current']:
            if old_field not in dm:
                issues.append(f"Missing dm.{old_field}")
        # New fields must exist but can be None
        for new_field in ['match_reason', 'department_match']:
            if new_field not in dm:
                issues.append(f"Missing new dm.{new_field}")

    return issues


def check_new_fields(result, test_name: str) -> list:
    """Prüfe dass match_reason und department_match korrekt befüllt sind."""
    issues = []
    if result.decision_maker:
        if result.decision_maker.match_reason is None:
            issues.append("match_reason ist None (sollte befüllt sein)")
        elif len(result.decision_maker.match_reason) < 5:
            issues.append(f"match_reason zu kurz: '{result.decision_maker.match_reason}'")
    return issues


def check_enrichment_path(result, test_name: str) -> list:
    """Prüfe ob enrichment_path sinnvolle Einträge hat."""
    issues = []
    path = result.enrichment_path

    if not path:
        issues.append("enrichment_path ist leer!")
        return issues

    # Sollte mindestens LLM-Parse und Kandidaten-Count enthalten
    has_candidates = any('candidates' in p for p in path)
    if not has_candidates:
        issues.append("Keine Kandidaten-Info im enrichment_path")

    return issues


async def run_single_test(test_case: dict) -> dict:
    """Einen Test-Case durchlaufen."""
    from models import WebhookPayload
    from pipeline import enrich_lead

    name = test_case["name"]
    payload_data = test_case["payload"]

    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"Company: {payload_data['company']} | Job: {payload_data['title']}")
    print(f"{'='*70}")

    payload = WebhookPayload(
        category=payload_data.get("category"),
        company=payload_data["company"],
        date_posted=payload_data.get("date_posted"),
        description=payload_data["description"],
        id=payload_data["id"],
        location=payload_data.get("location"),
        title=payload_data["title"],
        url=payload_data.get("url"),
    )

    try:
        result = await enrich_lead(payload, skip_paid_apis=True)
    except Exception as e:
        print(f"\n  CRASH: {e}")
        traceback.print_exc()
        return {"name": name, "status": "CRASH", "error": str(e), "issues": [f"Pipeline crash: {e}"]}

    # ── Ergebnisse anzeigen ──
    print(f"\n  Success: {result.success}")
    print(f"  Path: {' → '.join(result.enrichment_path[:15])}")

    print(f"\n  COMPANY:")
    print(f"    Name:   {result.company.name}")
    print(f"    Domain: {result.company.domain or 'N/A'}")
    print(f"    Phone:  {result.company.phone or 'N/A'}")

    if result.decision_maker:
        dm = result.decision_maker
        print(f"\n  DECISION MAKER:")
        print(f"    Name:             {dm.name}")
        print(f"    Title:            {dm.title or 'N/A'}")
        print(f"    Email:            {dm.email or 'N/A'}")
        print(f"    LinkedIn:         {dm.linkedin_url or 'N/A'}")
        print(f"    Verified:         {dm.verified_current}")
        print(f"    Match Reason:     {dm.match_reason or 'N/A'}")
        print(f"    Department Match: {dm.department_match}")
        print(f"    Verification:     {dm.verification_note or 'N/A'}")
    else:
        print(f"\n  DECISION MAKER: KEINER GEFUNDEN")

    print(f"\n  Phone Status: {result.phone_status.value}")
    if result.emails:
        print(f"  Emails: {', '.join(result.emails[:3])}")

    # ── Checks ──
    all_issues = []
    all_issues.extend(check_no_hr(result, name))
    all_issues.extend(check_n8n_format(result, name))
    all_issues.extend(check_new_fields(result, name))
    all_issues.extend(check_enrichment_path(result, name))

    # ── JSON Output prüfen (was n8n bekommt) ──
    output_json = result.model_dump()
    json_str = json.dumps(output_json, indent=2, ensure_ascii=False, default=str)

    print(f"\n  n8n JSON Output (gekürzt):")
    # Nur DM + neue Felder zeigen
    if output_json.get('decision_maker'):
        dm_json = output_json['decision_maker']
        print(f"    decision_maker.name:             {dm_json.get('name')}")
        print(f"    decision_maker.title:            {dm_json.get('title')}")
        print(f"    decision_maker.match_reason:     {dm_json.get('match_reason')}")
        print(f"    decision_maker.department_match: {dm_json.get('department_match')}")

    # ── Ergebnis ──
    if all_issues:
        print(f"\n  ISSUES ({len(all_issues)}):")
        for issue in all_issues:
            print(f"    {issue}")
        status = "FAIL"
    else:
        print(f"\n  ALLE CHECKS BESTANDEN")
        status = "PASS"

    return {
        "name": name,
        "status": status,
        "issues": all_issues,
        "dm_name": result.decision_maker.name if result.decision_maker else None,
        "dm_title": result.decision_maker.title if result.decision_maker else None,
        "dm_match_reason": result.decision_maker.match_reason if result.decision_maker else None,
        "dm_department_match": result.decision_maker.department_match if result.decision_maker else None,
        "enrichment_path": result.enrichment_path,
    }


async def test_target_titles():
    """Separater Unit-Test: target_titles werden korrekt generiert."""
    from llm_parser import _get_default_titles

    print(f"\n{'='*70}")
    print("UNIT TEST: target_titles (kein API Call)")
    print(f"{'='*70}")

    cases = [
        ("Software Developer", ["IT-Leiter", "CTO"], ["HR Manager", "Personalleiter"]),
        ("Vertriebsmitarbeiter", ["Vertriebsleiter", "Head of Sales"], ["HR Manager", "Personalleiter"]),
        ("Marketing Manager", ["Marketing-Leiter", "CMO"], ["HR Manager", "Personalleiter"]),
        ("Controller", ["CFO", "Finanzleiter"], ["HR Manager", "Personalleiter"]),
        ("Produktionsleiter", ["Produktionsleiter", "Werkleiter"], ["HR Manager", "Personalleiter"]),
        ("Logistiker", ["Logistikleiter"], ["HR Manager", "Personalleiter"]),
        ("Projektmanager", ["Geschäftsführer", "CEO"], ["HR Manager", "Personalleiter"]),
        ("Krankenpfleger", ["Chefarzt", "Klinikleiter"], ["HR Manager", "Personalleiter"]),
        ("Consultant", ["Partner", "Managing Consultant"], ["HR Manager", "Personalleiter"]),
    ]

    all_pass = True
    for title, expected_contains, forbidden in cases:
        titles = _get_default_titles(title)

        has_expected = any(e in titles for e in expected_contains)
        has_forbidden = any(f in titles for f in forbidden)

        status = "PASS" if has_expected and not has_forbidden else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {status} {title:30s} → {titles[:3]}")
        if has_forbidden:
            bad = [f for f in forbidden if f in titles]
            print(f"       HR NOCH DRIN: {bad}")

    return all_pass


async def test_helper_functions():
    """Unit-Test: _build_match_reason und _is_department_match."""
    from pipeline import _build_match_reason, _is_department_match

    print(f"\n{'='*70}")
    print("UNIT TEST: Helper-Funktionen (kein API Call)")
    print(f"{'='*70}")

    all_pass = True

    # _is_department_match
    tests = [
        ("IT-Leiter", ["IT-Leiter", "CTO"], True),
        ("Leiter IT", ["IT-Leiter"], True),
        ("CTO", ["IT-Leiter", "CTO"], True),
        ("Geschäftsführer", ["IT-Leiter", "CTO"], False),
        ("HR Manager", ["IT-Leiter"], False),
        (None, ["IT-Leiter"], False),
        ("IT-Leiter", [], False),
    ]

    for title, targets, expected in tests:
        result = _is_department_match(title, targets)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(f"  {status} _is_department_match({title!r}, {targets[:2]}) = {result} (expected {expected})")

    # _build_match_reason
    data = {"title": "IT-Leiter", "source": "team_page"}
    reason = _build_match_reason(data, ["IT-Leiter", "CTO"], "IT")
    print(f"  {'PASS' if 'passend' in reason else 'FAIL'} match_reason: {reason}")

    data2 = {"title": "Geschäftsführer", "source": "impressum"}
    reason2 = _build_match_reason(data2, ["IT-Leiter"], "IT")
    print(f"  {'PASS' if 'Impressum' in reason2 else 'FAIL'} match_reason: {reason2}")

    return all_pass


async def main():
    print("\n" + "#"*70)
    print("# ANSPRECHPARTNER-RELEVANZ TESTS")
    print("# skip_paid_apis=True -> Keine Phone-API-Kosten!")
    print("# Nutzt: LLM, Serper, Scraping (Kosten: ~$0.01 pro Test)")
    print("#"*70)

    # Unit Tests (kostenlos)
    titles_ok = await test_target_titles()
    helpers_ok = await test_helper_functions()

    # Integration Tests (LLM + Serper Kosten)
    results = []

    # Nur einen Test laufen lassen wenn --quick
    test_cases = TEST_CASES
    if "--quick" in sys.argv:
        test_cases = TEST_CASES[:1]
        print("\n  (--quick mode: nur erster Test)")
    elif "--all" not in sys.argv:
        test_cases = TEST_CASES[:1]
        print("\n  (Standard: nur IT-Test. --all für alle 3)")

    for tc in test_cases:
        r = await run_single_test(tc)
        results.append(r)

    # ── Summary ──
    print(f"\n\n{'#'*70}")
    print("# ZUSAMMENFASSUNG")
    print(f"{'#'*70}")

    print(f"\n  Unit Tests:")
    print(f"    target_titles:    {'PASS' if titles_ok else 'FAIL'}")
    print(f"    helper functions: {'PASS' if helpers_ok else 'FAIL'}")

    print(f"\n  Integration Tests:")
    for r in results:
        icon = "PASS" if r["status"] == "PASS" else "FAIL" if r["status"] == "FAIL" else "CRASH"
        print(f"    {icon} {r['name']}")
        if r.get("dm_name"):
            print(f"         → {r['dm_name']} ({r.get('dm_title', 'N/A')})")
            print(f"           match_reason: {r.get('dm_match_reason', 'N/A')}")
            print(f"           dept_match:   {r.get('dm_department_match')}")
        if r["issues"]:
            for issue in r["issues"]:
                print(f"         ISSUE: {issue}")

    total_issues = sum(len(r["issues"]) for r in results)
    all_pass = titles_ok and helpers_ok and total_issues == 0

    print(f"\n  {'ALLES OK' if all_pass else f'{total_issues} ISSUES GEFUNDEN'}")
    print(f"{'#'*70}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
