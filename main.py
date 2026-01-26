import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from config import get_settings
from models import WebhookPayload, EnrichmentResult
from pipeline import enrich_lead, enrich_lead_test_mode
from utils.stats import get_stats, get_stats_summary, reset_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lead Enrichment Service",
    description="Enriches job postings with decision maker contact info (Phone + Email)",
    version="1.0.0"
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/stats")
async def get_enrichment_stats():
    """
    Get enrichment service statistics.
    Shows success rates for Kaspr, FullEnrich, and country distribution.
    """
    return get_stats()


@app.get("/stats/summary")
async def get_enrichment_stats_summary():
    """
    Get human-readable stats summary.
    """
    return PlainTextResponse(content=get_stats_summary())


@app.post("/stats/reset")
async def reset_enrichment_stats():
    """
    Reset all statistics.
    """
    reset_stats()
    return {"status": "reset", "message": "Statistics have been reset"}


@app.post("/webhook/enrich")
async def webhook_enrich(
    payload: WebhookPayload,
    background_tasks: BackgroundTasks,
    test_mode: bool = Query(False, description="Skip paid APIs for testing")
):
    """
    Receives job posting from n8n and triggers enrichment pipeline.
    Returns immediately with job_id, enrichment runs in background.
    """
    logger.info(f"Received job posting: {payload.company} - {payload.title}")

    if test_mode:
        background_tasks.add_task(process_enrichment_test, payload)
    else:
        background_tasks.add_task(process_enrichment, payload)

    return {
        "status": "accepted",
        "job_id": payload.id,
        "message": f"Enrichment started for {payload.company}",
        "test_mode": test_mode
    }


@app.post("/webhook/enrich/sync", response_model=EnrichmentResult)
async def webhook_enrich_sync(
    payload: WebhookPayload,
    test_mode: bool = Query(False, description="Skip paid APIs for testing")
):
    """
    Synchronous version - waits for enrichment to complete.
    Use this for testing or when you need immediate results.

    Query params:
    - test_mode: Only use LLM + Impressum (no paid APIs)
    """
    logger.info(f"Received job posting (sync): {payload.company} - {payload.title}")
    logger.info(f"Options: test_mode={test_mode}")

    try:
        if test_mode:
            result = await enrich_lead_test_mode(payload)
        else:
            result = await enrich_lead(payload)
        return result
    except Exception as e:
        logger.error(f"Enrichment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/webhook/enrich/test", response_model=EnrichmentResult)
async def webhook_enrich_test(payload: WebhookPayload):
    """
    Test endpoint - only uses LLM parsing and free services.
    NO paid API credits consumed (FullEnrich, Kaspr skipped).
    """
    logger.info(f"TEST MODE: {payload.company} - {payload.title}")

    try:
        result = await enrich_lead_test_mode(payload)
        return result
    except Exception as e:
        logger.error(f"Test enrichment failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_enrichment(payload: WebhookPayload):
    """Background task to process enrichment."""
    try:
        result = await enrich_lead(payload)
        logger.info(f"Enrichment completed for {payload.company}: success={result.success}")

        if result.phone:
            logger.info(f"Found phone via {result.phone.source.value}")
        if result.decision_maker:
            logger.info(f"Decision maker: {result.decision_maker.name}")
        if result.emails:
            logger.info(f"Found {len(result.emails)} email(s)")

        # TODO: Send result to webhook callback or store in DB

    except Exception as e:
        logger.error(f"Background enrichment failed for {payload.company}: {e}", exc_info=True)


async def process_enrichment_test(payload: WebhookPayload):
    """Background task for test mode enrichment."""
    try:
        result = await enrich_lead_test_mode(payload)
        logger.info(f"TEST enrichment completed for {payload.company}: success={result.success}")
    except Exception as e:
        logger.error(f"TEST enrichment failed for {payload.company}: {e}", exc_info=True)


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
