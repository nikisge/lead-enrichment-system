from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # API Keys - Primary and Fallback
    anthropic_api_key: str = ""  # Primary key
    anthropic_api_key_fallback: str = ""  # Fallback key if primary fails
    apollo_api_key: str = ""
    kaspr_api_key: str = ""
    fullenrich_api_key: str = ""
    bettercontact_api_key: str = ""
    apify_api_key: str = ""  # For LinkedIn employment verification

    # OpenRouter for multi-model LLM access
    openrouter_api_key: str = ""

    # Optional: Google Custom Search
    google_api_key: str = ""
    google_cse_id: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Timeouts
    api_timeout: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
