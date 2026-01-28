"""
API Cost Tracking for Lead Enrichment Pipeline.

Tracks estimated costs for all API calls and provides summary logging.

Uses contextvars for thread-safe per-request tracking.
"""

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# Estimated costs per API call (in USD) - Updated January 2026
# Claude 4.5 pricing (Anthropic)
API_COSTS = {
    # Claude Sonnet 4.5 (via Anthropic API)
    "claude_sonnet_input": 0.003,      # $3/1M input tokens
    "claude_sonnet_output": 0.015,     # $15/1M output tokens

    # Claude Haiku 4.5 (via Anthropic API) - very cheap!
    "claude_haiku_input": 0.001,       # $1/1M input tokens
    "claude_haiku_output": 0.005,      # $5/1M output tokens

    # OpenRouter (Claude Haiku 4.5) - slight markup
    "openrouter_haiku_input": 0.001,   # ~$1/1M input tokens
    "openrouter_haiku_output": 0.005,  # ~$5/1M output tokens

    # Search APIs
    # Google CSE: Erste 100 Queries/Tag GRATIS, danach $5/1000
    # Wir tracken trotzdem für Übersicht, zeigen aber 0 Kosten an
    "google_cse": 0.0,                 # Gratis (erste 100/Tag)

    # Enrichment APIs (estimated, varies by plan)
    "bettercontact": 0.15,             # ~$0.15 per enrichment
    "fullenrich": 0.20,                # ~$0.20 per enrichment (avg)
    "kaspr": 0.10,                     # ~€0.10 per credit

    # Verification APIs
    "apify_linkedin": 0.004,           # ~$4/1000 profiles (HarvestAPI)
}

# Estimated tokens per LLM call type
LLM_TOKEN_ESTIMATES = {
    "job_parse": (2000, 500),          # (input, output)
    "team_rank": (1000, 200),
    "contact_extract": (3000, 300),
    "candidate_validate": (1500, 400),
    "company_research": (2000, 500),
    "email_match": (200, 50),
    "linkedin_match": (200, 50),
    "impressum_extract": (2000, 300),
}


@dataclass
class APICall:
    """Record of a single API call."""
    api_name: str
    call_type: str
    estimated_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    details: str = ""


@dataclass
class CostSummary:
    """Summary of costs for an enrichment run."""
    total_cost: float = 0.0
    calls_by_api: Dict[str, int] = field(default_factory=dict)
    costs_by_api: Dict[str, float] = field(default_factory=dict)
    call_details: List[APICall] = field(default_factory=list)


class CostTracker:
    """
    Tracks API costs for a single enrichment run.

    Usage:
        tracker = CostTracker()
        tracker.track_llm_call("job_parse", tier="sonnet")
        tracker.track_google_search()
        tracker.track_enrichment_api("kaspr")
        tracker.log_summary()
    """

    def __init__(self, company_name: str = ""):
        self.company_name = company_name
        self.calls: List[APICall] = []
        self.start_time = datetime.now()

    def track_llm_call(
        self,
        call_type: str,
        tier: str = "haiku",
        success: bool = True
    ):
        """
        Track an LLM API call.

        Args:
            call_type: Type of call (job_parse, contact_extract, etc.)
            tier: Model tier (sonnet, haiku)
            success: Whether the call succeeded
        """
        # Get token estimates
        input_tokens, output_tokens = LLM_TOKEN_ESTIMATES.get(
            call_type, (500, 200)
        )

        # Calculate cost based on tier (Claude 4.5 models)
        if tier == "sonnet":
            cost = (
                (input_tokens / 1000) * API_COSTS["claude_sonnet_input"] +
                (output_tokens / 1000) * API_COSTS["claude_sonnet_output"]
            )
            api_name = "Claude Sonnet 4.5"
        else:
            cost = (
                (input_tokens / 1000) * API_COSTS["claude_haiku_input"] +
                (output_tokens / 1000) * API_COSTS["claude_haiku_output"]
            )
            api_name = "Claude Haiku 4.5"

        self.calls.append(APICall(
            api_name=api_name,
            call_type=call_type,
            estimated_cost=cost,
            success=success,
            details=f"{input_tokens}+{output_tokens} tokens"
        ))

    def track_openrouter_call(
        self,
        call_type: str,
        success: bool = True
    ):
        """Track an OpenRouter API call (Claude Haiku 4.5)."""
        input_tokens, output_tokens = LLM_TOKEN_ESTIMATES.get(
            call_type, (500, 200)
        )

        cost = (
            (input_tokens / 1000) * API_COSTS["openrouter_haiku_input"] +
            (output_tokens / 1000) * API_COSTS["openrouter_haiku_output"]
        )

        self.calls.append(APICall(
            api_name="OpenRouter (Haiku 4.5)",
            call_type=call_type,
            estimated_cost=cost,
            success=success,
            details=f"{input_tokens}+{output_tokens} tokens"
        ))

    def track_google_search(self, query_type: str = "search"):
        """Track a Google Custom Search API call (free tier: 100/day)."""
        self.calls.append(APICall(
            api_name="Google CSE (gratis)",
            call_type=query_type,
            estimated_cost=API_COSTS["google_cse"],
            details="1 query (100/Tag gratis)"
        ))

    def track_enrichment_api(
        self,
        api_name: str,
        success: bool = True,
        found_phone: bool = False,
        found_email: bool = False
    ):
        """
        Track an enrichment API call (BetterContact, FullEnrich, Kaspr).

        Note: Some APIs only charge on success, others charge per call.
        """
        cost_key = api_name.lower().replace(" ", "")
        cost = API_COSTS.get(cost_key, 0.10)

        details = []
        if found_phone:
            details.append("phone found")
        if found_email:
            details.append("email found")
        if not success:
            details.append("no result")
            # These APIs don't charge when no result found
            if api_name.lower() in ["bettercontact", "fullenrich"]:
                cost = 0.0

        self.calls.append(APICall(
            api_name=api_name,
            call_type="enrichment",
            estimated_cost=cost,
            success=success,
            details=", ".join(details) if details else "called"
        ))

    def track_apify_scrape(self, success: bool = True):
        """Track an Apify LinkedIn profile scrape."""
        self.calls.append(APICall(
            api_name="Apify",
            call_type="linkedin_verify",
            estimated_cost=API_COSTS["apify_linkedin"],
            success=success,
            details="profile scrape"
        ))

    def get_summary(self) -> CostSummary:
        """Get cost summary for this run."""
        summary = CostSummary()

        for call in self.calls:
            # Total cost
            summary.total_cost += call.estimated_cost

            # Calls by API
            if call.api_name not in summary.calls_by_api:
                summary.calls_by_api[call.api_name] = 0
                summary.costs_by_api[call.api_name] = 0.0

            summary.calls_by_api[call.api_name] += 1
            summary.costs_by_api[call.api_name] += call.estimated_cost

        summary.call_details = self.calls
        return summary

    def log_summary(self):
        """Log a formatted cost summary."""
        summary = self.get_summary()
        duration = (datetime.now() - self.start_time).total_seconds()

        # Build summary lines
        lines = [
            f"",
            f"{'='*60}",
            f"COST SUMMARY: {self.company_name}",
            f"{'='*60}",
            f"Duration: {duration:.1f}s",
            f"Total API Calls: {len(self.calls)}",
            f"Estimated Total Cost: ${summary.total_cost:.4f}",
            f"",
            f"Breakdown by API:",
        ]

        for api_name, count in sorted(summary.calls_by_api.items()):
            cost = summary.costs_by_api[api_name]
            lines.append(f"  - {api_name}: {count} calls = ${cost:.4f}")

        lines.append(f"")
        lines.append(f"Call Details:")

        for call in self.calls:
            status = "✓" if call.success else "✗"
            lines.append(
                f"  {status} {call.api_name} ({call.call_type}): "
                f"${call.estimated_cost:.4f} [{call.details}]"
            )

        lines.append(f"{'='*60}")

        # Log as single block
        logger.info("\n".join(lines))

        return summary

    def get_cost_line(self) -> str:
        """Get a single-line cost summary for inline logging."""
        summary = self.get_summary()
        return f"[Cost: ${summary.total_cost:.4f} | {len(self.calls)} API calls]"


# Thread-safe per-request tracker using contextvars
_current_tracker: ContextVar[Optional[CostTracker]] = ContextVar('cost_tracker', default=None)


def start_cost_tracking(company_name: str = "") -> CostTracker:
    """Start tracking costs for a new enrichment run (thread-safe)."""
    tracker = CostTracker(company_name)
    _current_tracker.set(tracker)
    return tracker


def get_cost_tracker() -> Optional[CostTracker]:
    """Get the current cost tracker (thread-safe)."""
    return _current_tracker.get()


def track_llm(call_type: str, tier: str = "haiku", success: bool = True):
    """Convenience function to track LLM call."""
    tracker = _current_tracker.get()
    if tracker:
        tracker.track_llm_call(call_type, tier, success)


def track_openrouter(call_type: str, success: bool = True):
    """Convenience function to track OpenRouter call."""
    tracker = _current_tracker.get()
    if tracker:
        tracker.track_openrouter_call(call_type, success)


def track_google(query_type: str = "search"):
    """Convenience function to track Google search."""
    tracker = _current_tracker.get()
    if tracker:
        tracker.track_google_search(query_type)


def track_enrichment(
    api_name: str,
    success: bool = True,
    found_phone: bool = False,
    found_email: bool = False
):
    """Convenience function to track enrichment API."""
    tracker = _current_tracker.get()
    if tracker:
        tracker.track_enrichment_api(api_name, success, found_phone, found_email)


def track_apify(success: bool = True):
    """Convenience function to track Apify call."""
    tracker = _current_tracker.get()
    if tracker:
        tracker.track_apify_scrape(success)


def log_cost_summary():
    """Log the cost summary for current run."""
    tracker = _current_tracker.get()
    if tracker:
        return tracker.log_summary()
    return None
