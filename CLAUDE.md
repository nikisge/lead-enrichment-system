# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Service

```bash
# Local development
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Docker
docker-compose up              # runs on port 8070 (maps to internal 8000)
docker build -t lead-enrichment .

# After fresh install: Playwright needs browser binaries
pip install -r requirements.txt
playwright install chromium
```

## Testing

Tests are standalone scripts (no pytest runner configured):
```bash
python test_full_flow.py          # Full pipeline integration test
python test_apis.py               # Individual API client tests
python test_apify_linkedin.py     # LinkedIn verification
python test_dm_discovery.py       # Decision maker discovery
python test_team_discovery.py     # Team page discovery
python test_linkedin_search.py    # LinkedIn search
python test_local.py              # Local dev tests (free APIs only)
python test_local.py --full       # Local dev tests (includes paid APIs)
```

## Endpoints

- `POST /webhook/enrich/sync` — Main enrichment endpoint (n8n calls this, 8 min timeout). Supports `?test_mode=true` to skip paid APIs
- `POST /webhook/enrich` — Async version (result storage not implemented)
- `POST /webhook/enrich/test` — Test-only endpoint, never uses paid APIs
- `GET /health` — Health check
- `GET /stats` / `/stats/summary` — Phone enrichment service stats
- `GET /dashboard` / `/dashboard/summary` — Pipeline quality dashboard
- `GET /dashboard/recent?n=60` — Last N pipeline runs as JSON
- `POST /stats/reset` — Reset all stats

## Architecture

FastAPI service that enriches job postings with decision maker contact info (phone + email) through a 7-phase AI pipeline orchestrated in `pipeline.py`:

1. **LLM Parse** (`llm_parser.py`) — Extract company/contact from job posting text using Claude Sonnet (direct Anthropic API with fallback key)
2. **Domain Discovery** (`pipeline.py`) — 7-tier priority: Job URL → LLM-extracted → Serper.dev ($0.001/query) → Google Knowledge Graph (free, 100K/day) → DuckDuckGo (free) → Heuristic (free) → Google CSE (100/day limit). Validates via AI + parked domain detection
3. **Parallel Scraping** — Simultaneous via `asyncio.gather`: job URL scraping, impressum scraping (Playwright), team page discovery (sitemap + pattern matching + AI)
4. **Candidate Collection & Ranking** — 5 sources ranked by trust (job_url=100, llm_parse=90, team_page=70, impressum=50, linkedin_fallback=40). AI validates and deduplicates, returns top 3
5. **LinkedIn Verification** (`apify_linkedin.py`) — Apify/HarvestAPI confirms current employment. Untrusted sources (linkedin_fallback) are dropped if unverified
6. **Phone Enrichment** — Waterfall: FullEnrich → Kaspr (Kaspr only for 1st candidate, costs per request). Up to 2 candidates tried. BetterContact deactivated
7. **Company Research** (`company_research.py`) — Runs in parallel with phone enrichment via `asyncio.create_task`

The entire pipeline is wrapped in `asyncio.wait_for()` with a 110s internal timeout. On timeout, partial results are returned.

## Debugging Pipeline Runs

`EnrichmentResult.enrichment_path` is a list of strings tracing each pipeline step (e.g., `["llm_parse: found company X", "domain: google_cse → example.com", ...]`). Use this to trace which phases ran and what they produced.

`EnrichmentResult.warnings` flags issues like `"used_fallback_api_key"`. `operational_alerts` is a dict of booleans for n8n monitoring (e.g., `"anthropic_key_missing": true`).

## LLM Strategy

- **OpenRouter** (`clients/llm_client.py`): Gemini Flash (fast/cheap), Haiku (balanced), Sonnet (smart) — used for extractions, validations, ranking
- **Direct Anthropic API** (`llm_parser.py`, `company_research.py`): Sonnet for job parsing and company research — these bypass llm_client

## Trust Model

Sources `job_url`, `llm_parse`, `team_page`, `impressum` are trusted (kept without LinkedIn verification). Source `linkedin_fallback` (from DM Google search) is untrusted and MUST be verified via Apify before use.

## Cost-Sensitive APIs

- **Kaspr**: Charges per request (1 credit even with no result) — only use for 1st candidate with verified LinkedIn
- **BetterContact / FullEnrich**: Free on no-result (pay per successful result only), polling capped at 6 attempts / 30s
- **Apollo**: Returns 403 on free plan — do NOT integrate
- **Google CSE**: 100 free queries/day — DM fallback consumes additional quota

## Phone Validation

`_is_valid_dach_phone()` in pipeline.py: accepts only DACH numbers (+49/+43/+41), blocks service numbers (0800/0900/0180/0137/0700/0190/0191), requires 10-15 digits.

## Key Modules

- `pipeline.py` (~2000 lines): Main orchestration — domain discovery, scraping, candidate ranking, phone enrichment waterfall, all pipeline phases. Internal pipeline timeout is 110s.
- `llm_parser.py`: Job posting parsing (direct Anthropic API, not OpenRouter)
- `models.py`: All Pydantic models — `WebhookPayload` (input), `EnrichmentResult` (output), `DecisionMaker`, `PhoneResult`, `CompanyIntel`, etc.
- `clients/`:
  - `llm_client.py` — OpenRouter multi-model LLM (shared by most AI calls)
  - `ai_extractor.py` — AI-based contact extraction from page HTML
  - `ai_validator.py` — AI-based name/email/candidate validation
  - `team_discovery.py` — Sitemap + pattern matching for team/about pages
  - `apify_linkedin.py` — LinkedIn employment verification via HarvestAPI
  - `impressum.py` — Playwright-based Impressum page scraping
  - `kaspr.py` / `bettercontact.py` / `fullenrich.py` — Phone enrichment APIs
  - `company_research.py` — Sales brief generation (direct Anthropic API)
- `utils/`: `phone.py` (formatting via `phonenumbers`), `stats.py` (JSON-file persistence), `cost_tracker.py` (per-request tracking via `ContextVar`)

## Code Conventions

- **Client factories**: Each client module exposes `get_*_client()` (e.g., `get_kaspr_client()`) returning a fresh instance
- **HTTP clients**: Created per-request with `async with httpx.AsyncClient()` — no long-lived connections
- **Error handling**: Graceful degradation — functions return `None` or empty list on failure, never crash the pipeline. Exceptions are logged with `exc_info=True`
- **Logging**: Module-level `logger = logging.getLogger(__name__)`. Info for milestones, warning for fallbacks, error for exceptions
- **Pipeline language**: German in user-facing output (enrichment_path, company research). Code and logs in English

## Configuration

All config via `.env` file loaded by `config.py` (Pydantic BaseSettings with `@lru_cache`). Required keys: `ANTHROPIC_API_KEY`, `ANTHROPIC_API_KEY_FALLBACK`, `OPENROUTER_API_KEY`, `APIFY_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_CSE_ID`, and phone enrichment keys (`KASPR_API_KEY`, `FULLENRICH_API_KEY`, `BETTERCONTACT_API_KEY`). Optional: `SERPER_API_KEY` for Serper.dev Google SERP domain discovery.

## Deployment

Docker on `91.99.149.237:8070`. Persistent volume at `/app/data` stores `enrichment_stats.json` and `pipeline_stats.json`.
