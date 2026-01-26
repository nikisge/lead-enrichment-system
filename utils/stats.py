"""
Statistics tracking for enrichment services.
Tracks success rates for Kaspr, FullEnrich, and other services.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from threading import Lock

logger = logging.getLogger(__name__)

# File path for stats storage
STATS_FILE = Path(__file__).parent.parent / "enrichment_stats.json"
_file_lock = Lock()


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
        "returned_phones": 0,           # API returned at least one phone
        "dach_valid_phones": 0,         # At least one phone passed DACH filter
        "mobile_found": 0,              # Mobile phone found
        "landline_found": 0,            # Only landline found
        "filtered_out": 0,              # Phones returned but all filtered (non-DACH)
        "no_phone_returned": 0,         # API returned no phones at all
        "phone_countries": {},          # Country codes found (for analysis)
        "last_success": None,
        "last_attempt": None,
    }


def track_phone_attempt(
    service: str,
    phones_returned: List[Any],
    dach_valid_phone: Optional[Any],
    phone_type: Optional[str] = None
) -> None:
    """
    Track a phone enrichment attempt.

    Args:
        service: Service name ('kaspr' or 'fullenrich')
        phones_returned: List of phone results from the API
        dach_valid_phone: The phone that passed DACH filter (or None)
        phone_type: Type of the valid phone ('mobile', 'landline', 'unknown')
    """
    with _file_lock:
        stats = _load_stats()

        if service not in stats["services"]:
            stats["services"][service] = _get_default_service_stats()

        svc = stats["services"][service]
        svc["total_attempts"] += 1
        svc["last_attempt"] = datetime.now().isoformat()

        if phones_returned:
            svc["returned_phones"] += 1

            # Track country codes for analysis
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

    # Clean number
    cleaned = number.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

    # Check for + prefix
    if cleaned.startswith("+"):
        # Common country codes
        if cleaned.startswith("+49"):
            return "DE"
        elif cleaned.startswith("+43"):
            return "AT"
        elif cleaned.startswith("+41"):
            return "CH"
        elif cleaned.startswith("+1"):
            return "US/CA"
        elif cleaned.startswith("+44"):
            return "UK"
        elif cleaned.startswith("+33"):
            return "FR"
        elif cleaned.startswith("+31"):
            return "NL"
        elif cleaned.startswith("+32"):
            return "BE"
        elif cleaned.startswith("+39"):
            return "IT"
        elif cleaned.startswith("+34"):
            return "ES"
        elif cleaned.startswith("+48"):
            return "PL"
        elif cleaned.startswith("+420"):
            return "CZ"
        else:
            # Extract first 2-3 digits as code
            return f"+{cleaned[1:4]}"

    # Check for 00 prefix
    if cleaned.startswith("00"):
        if cleaned.startswith("0049"):
            return "DE"
        elif cleaned.startswith("0043"):
            return "AT"
        elif cleaned.startswith("0041"):
            return "CH"
        return f"00{cleaned[2:5]}"

    # German national format
    if cleaned.startswith("0"):
        return "DE (national)"

    return "unknown"


def get_stats() -> Dict[str, Any]:
    """Get current statistics."""
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
            sorted_countries = sorted(countries.items(), key=lambda x: x[1], reverse=True)
            for country, count in sorted_countries[:10]:
                lines.append(f"    {country}: {count}")
        else:
            lines.append("    (no data yet)")

        lines.append("")

    return "\n".join(lines)


def reset_stats() -> None:
    """Reset all statistics."""
    with _file_lock:
        _save_stats(_get_default_stats())
        logger.info("Statistics reset")
