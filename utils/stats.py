"""
Statistics tracking for enrichment services.
Tracks pipeline success rates and phone service stats.
Persists to /app/data/ volume (survives container restarts).
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

# Persistent data directory (Docker volume mount)
DATA_DIR = Path("/app/data")
if not DATA_DIR.exists():
    # Local dev fallback
    DATA_DIR = Path(__file__).parent.parent / "data"
    DATA_DIR.mkdir(exist_ok=True)

STATS_FILE = DATA_DIR / "enrichment_stats.json"
PIPELINE_STATS_FILE = DATA_DIR / "pipeline_stats.json"
_file_lock = Lock()


# ========== PHONE SERVICE STATS (existing) ==========

def _load_stats() -> Dict[str, Any]:
    """Load stats from JSON file."""
    if not STATS_FILE.exists():
        return _get_default_stats()
    try:
        with open(STATS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load stats file: {e}")
        return _get_default_stats()


def _save_stats(stats: Dict[str, Any]) -> None:
    """Save stats to JSON file."""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    except IOError as e:
        logger.warning(f"Could not save stats file: {e}")


def _get_default_stats() -> Dict[str, Any]:
    """Return default stats structure."""
    return {
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "services": {
            "kaspr": _get_default_service_stats(),
            "fullenrich": _get_default_service_stats(),
        }
    }


def _get_default_service_stats() -> Dict[str, Any]:
    """Return default stats for a single service."""
    return {
        "total_attempts": 0,
        "returned_phones": 0,
        "dach_valid_phones": 0,
        "mobile_found": 0,
        "landline_found": 0,
        "filtered_out": 0,
        "no_phone_returned": 0,
        "phone_countries": {},
        "last_success": None,
        "last_attempt": None,
    }


def track_phone_attempt(
    service: str,
    phones_returned: List[Any],
    dach_valid_phone: Optional[Any],
    phone_type: Optional[str] = None
) -> None:
    """Track a phone enrichment attempt."""
    with _file_lock:
        stats = _load_stats()

        if service not in stats["services"]:
            stats["services"][service] = _get_default_service_stats()

        svc = stats["services"][service]
        svc["total_attempts"] += 1
        svc["last_attempt"] = datetime.now().isoformat()

        if phones_returned:
            svc["returned_phones"] += 1

            for phone in phones_returned:
                number = phone.number if hasattr(phone, 'number') else str(phone)
                country = _extract_country_code(number)
                if country:
                    svc["phone_countries"][country] = svc["phone_countries"].get(country, 0) + 1

            if dach_valid_phone:
                svc["dach_valid_phones"] += 1
                svc["last_success"] = datetime.now().isoformat()
                if phone_type == "mobile":
                    svc["mobile_found"] += 1
                elif phone_type == "landline":
                    svc["landline_found"] += 1
            else:
                svc["filtered_out"] += 1
        else:
            svc["no_phone_returned"] += 1

        stats["last_updated"] = datetime.now().isoformat()
        _save_stats(stats)


def _extract_country_code(number: str) -> Optional[str]:
    """Extract country code from phone number for statistics."""
    if not number:
        return None
    cleaned = number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    if cleaned.startswith("+"):
        if cleaned.startswith("+49"): return "DE"
        elif cleaned.startswith("+43"): return "AT"
        elif cleaned.startswith("+41"): return "CH"
        elif cleaned.startswith("+1"): return "US/CA"
        elif cleaned.startswith("+44"): return "UK"
        elif cleaned.startswith("+33"): return "FR"
        elif cleaned.startswith("+31"): return "NL"
        elif cleaned.startswith("+32"): return "BE"
        elif cleaned.startswith("+39"): return "IT"
        elif cleaned.startswith("+34"): return "ES"
        elif cleaned.startswith("+48"): return "PL"
        elif cleaned.startswith("+420"): return "CZ"
        else: return f"+{cleaned[1:4]}"
    if cleaned.startswith("00"):
        if cleaned.startswith("0049"): return "DE"
        elif cleaned.startswith("0043"): return "AT"
        elif cleaned.startswith("0041"): return "CH"
        return f"00{cleaned[2:5]}"
    if cleaned.startswith("0"):
        return "DE (national)"
    return "unknown"


def get_stats() -> Dict[str, Any]:
    """Get current statistics (phone services)."""
    with _file_lock:
        return _load_stats()


def get_stats_summary() -> str:
    """Get a human-readable stats summary."""
    stats = get_stats()
    lines = [
        "=" * 60,
        "ENRICHMENT SERVICE STATISTICS",
        "=" * 60,
        f"Last updated: {stats.get('last_updated', 'N/A')}",
        ""
    ]
    for service_name, svc in stats.get("services", {}).items():
        total = svc.get("total_attempts", 0)
        returned = svc.get("returned_phones", 0)
        dach_valid = svc.get("dach_valid_phones", 0)
        mobile = svc.get("mobile_found", 0)
        filtered = svc.get("filtered_out", 0)
        success_rate = (dach_valid / total * 100) if total > 0 else 0
        mobile_rate = (mobile / dach_valid * 100) if dach_valid > 0 else 0
        filter_rate = (filtered / returned * 100) if returned > 0 else 0
        lines.extend([
            f"--- {service_name.upper()} ---",
            f"  Total attempts:     {total}",
            f"  Returned phones:    {returned} ({returned/total*100:.1f}% of attempts)" if total > 0 else f"  Returned phones:    {returned}",
            f"  DACH valid:         {dach_valid} ({success_rate:.1f}% success rate)",
            f"  Mobile found:       {mobile} ({mobile_rate:.1f}% of valid)",
            f"  Filtered out:       {filtered} ({filter_rate:.1f}% non-DACH)",
            f"  No phone returned:  {svc.get('no_phone_returned', 0)}",
            f"  Last success:       {svc.get('last_success', 'Never')}",
            "",
            f"  Country distribution:",
        ])
        countries = svc.get("phone_countries", {})
        if countries:
            for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"    {country}: {count}")
        else:
            lines.append("    (no data yet)")
        lines.append("")
    return "\n".join(lines)


def reset_stats() -> None:
    """Reset phone service statistics."""
    with _file_lock:
        _save_stats(_get_default_stats())
        logger.info("Phone service statistics reset")


# ========== PIPELINE STATS (new - overall quality tracking) ==========

def _load_pipeline_stats() -> Dict[str, Any]:
    """Load pipeline stats from JSON file."""
    if not PIPELINE_STATS_FILE.exists():
        return _get_default_pipeline_stats()
    try:
        with open(PIPELINE_STATS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not load pipeline stats: {e}")
        return _get_default_pipeline_stats()


def _save_pipeline_stats(stats: Dict[str, Any]) -> None:
    """Save pipeline stats to JSON file."""
    try:
        with open(PIPELINE_STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    except IOError as e:
        logger.warning(f"Could not save pipeline stats: {e}")


def _get_default_pipeline_stats() -> Dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "total_runs": 0,
        "domain_found": 0,
        "domain_source": {},
        "decision_maker_found": 0,
        "linkedin_verified": 0,
        "phone_found": 0,
        "phone_source": {},
        "email_found": 0,
        "company_phone_found": 0,
        "company_address_found": 0,
        "company_research_done": 0,
        "timeouts": 0,
        "recent_runs": [],
    }


def track_pipeline_result(result) -> None:
    """
    Track a complete pipeline run result for quality monitoring.
    Called after each enrichment completes.
    """
    with _file_lock:
        stats = _load_pipeline_stats()
        stats["total_runs"] += 1
        stats["last_updated"] = datetime.now().isoformat()

        path = result.enrichment_path

        # Domain found?
        has_domain = result.company.domain is not None
        if has_domain:
            stats["domain_found"] += 1
        # Domain source (prefix-match for new enrichment_path format: "domain:source->...")
        domain_source_prefixes = {
            "domain:job_url->": "job_url",
            "domain:llm_validated->": "llm_validated",
            "domain:serper->": "serper",
            "domain:ddg->": "ddg",
            "domain:kg->": "kg",
            "domain:google_cse->": "google_cse",
            "domain:heuristic->": "heuristic",
        }
        for entry in path:
            matched = False
            for prefix, source_key in domain_source_prefixes.items():
                if entry.startswith(prefix):
                    stats["domain_source"][source_key] = stats["domain_source"].get(source_key, 0) + 1
                    matched = True
                    break
            if matched:
                break

        # Decision maker?
        has_dm = result.decision_maker is not None
        if has_dm:
            stats["decision_maker_found"] += 1

        # LinkedIn verified?
        if any(p.startswith("linkedin_verified_") for p in path):
            stats["linkedin_verified"] += 1

        # Phone found?
        has_phone = result.phone is not None
        if has_phone:
            stats["phone_found"] += 1
            source = result.phone.source.value
            stats["phone_source"][source] = stats["phone_source"].get(source, 0) + 1

        # Email found?
        has_email = len(result.emails) > 0
        if has_email:
            stats["email_found"] += 1

        # Company phone found?
        has_company_phone = result.company.phone is not None
        if has_company_phone:
            stats["company_phone_found"] = stats.get("company_phone_found", 0) + 1

        # Company address found?
        has_company_address = bool(result.company.address)
        if has_company_address:
            stats["company_address_found"] = stats.get("company_address_found", 0) + 1

        # Company research?
        if "company_research" in path:
            stats["company_research_done"] += 1

        # Timeout?
        if "pipeline_timeout" in path:
            stats["timeouts"] += 1

        # Recent runs log (keep last 100)
        stats["recent_runs"].append({
            "timestamp": datetime.now().isoformat(),
            "company": result.company.name,
            "domain": result.company.domain,
            "company_phone": result.company.phone,
            "company_address": bool(result.company.address),
            "dm_name": result.decision_maker.name if result.decision_maker else None,
            "dm_title": result.decision_maker.title if result.decision_maker else None,
            "dm_linkedin": result.decision_maker.linkedin_url if result.decision_maker else None,
            "dm_email": result.decision_maker.email if result.decision_maker else None,
            "dm_verified": result.decision_maker.verified_current if result.decision_maker else False,
            "phone_found": has_phone,
            "phone_number": result.phone.number if result.phone else None,
            "phone_source": result.phone.source.value if result.phone else None,
            "phone_status": result.phone_status.value,
            "emails_count": len(result.emails),
            "job_title": result.job_title,
            "success": result.success,
            "enrichment_path": path,
        })
        if len(stats["recent_runs"]) > 100:
            stats["recent_runs"] = stats["recent_runs"][-100:]

        _save_pipeline_stats(stats)


def get_pipeline_stats() -> Dict[str, Any]:
    """Get pipeline quality stats."""
    with _file_lock:
        return _load_pipeline_stats()


def get_pipeline_dashboard() -> str:
    """Human-readable pipeline quality dashboard."""
    stats = get_pipeline_stats()
    total = stats.get("total_runs", 0)
    if total == 0:
        return "No pipeline runs yet."

    def pct(n): return f"{n/total*100:.0f}%"

    domain = stats.get("domain_found", 0)
    dm = stats.get("decision_maker_found", 0)
    li_verified = stats.get("linkedin_verified", 0)
    phone = stats.get("phone_found", 0)
    email = stats.get("email_found", 0)
    company_phone = stats.get("company_phone_found", 0)
    company_address = stats.get("company_address_found", 0)
    research = stats.get("company_research_done", 0)
    timeouts = stats.get("timeouts", 0)

    lines = [
        "=" * 60,
        "PIPELINE QUALITY DASHBOARD",
        "=" * 60,
        f"Total Runs:           {total}",
        f"Last Updated:         {stats.get('last_updated', 'N/A')}",
        "",
        "--- KUNDEN-ÜBERSICHT ---",
        f"  Website/Domain:        {domain}/{total} ({pct(domain)})",
        f"  Firmentelefon (Impr.): {company_phone}/{total} ({pct(company_phone)})",
        f"  Ansprechpartner:       {dm}/{total} ({pct(dm)})",
        f"  Telefon DM (API):      {phone}/{total} ({pct(phone)})",
        f"  Tel. gesamt (Firma|DM): {company_phone + phone}/{total} ({pct(company_phone + phone)})",
        f"  Email Ansprechp.:      {email}/{total} ({pct(email)})",
        "",
        "--- DETAILS ---",
        f"  LinkedIn Verified:  {li_verified}/{total} ({pct(li_verified)})",
        f"  Firmenadresse:      {company_address}/{total} ({pct(company_address)})",
        f"  Company Research:   {research}/{total} ({pct(research)})",
        f"  Timeouts:           {timeouts}/{total} ({pct(timeouts)})",
        "",
        "--- DOMAIN SOURCE ---",
    ]
    source_labels = {
        "job_url": "Job URL",
        "llm_validated": "LLM Parser",
        "serper": "Serper.dev",
        "ddg": "DuckDuckGo",
        "kg": "Knowledge Graph",
        "google_cse": "Google CSE",
        "heuristic": "Heuristic",
    }
    for key, label in source_labels.items():
        count = stats.get("domain_source", {}).get(key, 0)
        if count > 0:
            lines.append(f"  {label}: {count} ({count/domain*100:.0f}%)" if domain > 0 else f"  {label}: {count}")

    lines.append("")
    lines.append("--- PHONE SOURCE ---")
    for source, count in sorted(stats.get("phone_source", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {source}: {count}")

    # Recent runs
    recent = stats.get("recent_runs", [])
    if recent:
        lines.append("")
        lines.append(f"--- LAST {min(len(recent), 10)} RUNS ---")
        for run in recent[-10:]:
            dm_phone = "TEL" if run.get("phone_found") else "---"
            co_phone = "FIR" if run.get("company_phone") else "---"
            dm_name = run.get("dm_name", "---") or "---"
            domain_val = run.get("domain", "---") or "---"
            company = run.get("company", "???")[:25]
            lines.append(f"  [{co_phone}|{dm_phone}] {company:<25} | {domain_val:<22} | {dm_name}")

    return "\n".join(lines)


def reset_pipeline_stats() -> None:
    """Reset pipeline statistics."""
    with _file_lock:
        _save_pipeline_stats(_get_default_pipeline_stats())
        logger.info("Pipeline statistics reset")
