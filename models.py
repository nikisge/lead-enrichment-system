from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
from enum import Enum


class PhoneSource(str, Enum):
    BETTERCONTACT = "bettercontact"
    KASPR = "kaspr"
    FULLENRICH = "fullenrich"
    IMPRESSUM = "impressum"
    COMPANY_MAIN = "company_main"


class PhoneType(str, Enum):
    MOBILE = "mobile"
    LANDLINE = "landline"
    UNKNOWN = "unknown"


# Input size limits to prevent abuse
MAX_DESCRIPTION_LENGTH = 50000  # 50KB max for job description
MAX_FIELD_LENGTH = 1000  # 1KB max for other fields


# Webhook Input (from n8n)
class WebhookPayload(BaseModel):
    category: Optional[str] = Field(default=None, max_length=MAX_FIELD_LENGTH)
    company: str = Field(..., max_length=MAX_FIELD_LENGTH)
    date_posted: Optional[str] = Field(default=None, max_length=100)
    description: str = Field(..., max_length=MAX_DESCRIPTION_LENGTH)
    id: str = Field(..., max_length=MAX_FIELD_LENGTH)
    location: Optional[str] = Field(default=None, max_length=MAX_FIELD_LENGTH)
    seen: Optional[bool] = False
    source: Optional[str] = Field(default=None, max_length=MAX_FIELD_LENGTH)
    title: str = Field(..., max_length=MAX_FIELD_LENGTH)
    url: Optional[str] = Field(default=None, max_length=2000)

    @field_validator('description')
    @classmethod
    def truncate_description(cls, v: str) -> str:
        """Truncate description if too long (graceful handling)."""
        if v and len(v) > MAX_DESCRIPTION_LENGTH:
            return v[:MAX_DESCRIPTION_LENGTH]
        return v


# LLM Parsing Result
class ParsedJobPosting(BaseModel):
    company_name: str
    company_domain: Optional[str] = None
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None  # Phone from job posting
    target_titles: List[str] = Field(default_factory=list)
    department: Optional[str] = None
    location: Optional[str] = None


# Decision Maker
class DecisionMaker(BaseModel):
    name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    title: Optional[str] = None
    linkedin_url: Optional[str] = None
    email: Optional[str] = None
    apollo_id: Optional[str] = None
    verified_current: bool = True  # False if we're not sure they still work there
    verification_note: Optional[str] = None  # e.g. "(nicht verifiziert - könnte nicht mehr dort arbeiten)"


# Phone Result
class PhoneResult(BaseModel):
    number: str
    type: PhoneType = PhoneType.UNKNOWN
    source: PhoneSource
    context_note: str = ""  # e.g. "Möglicherweise privat - von LinkedIn Profil"


# Company Info
class CompanyInfo(BaseModel):
    name: str
    domain: Optional[str] = None
    industry: Optional[str] = None
    employee_count: Optional[str] = None
    location: Optional[str] = None
    address: Optional[str] = None  # Street address from Impressum
    phone: Optional[str] = None
    website: Optional[str] = None
    linkedin_url: Optional[str] = None


# Company Intelligence for Sales
class CompanyIntel(BaseModel):
    """Company research data for sales preparation."""
    summary: str = ""  # AI-generated sales brief
    description: str = ""  # What the company does
    industry: str = ""
    employee_count: Optional[str] = None
    founded: Optional[str] = None
    headquarters: str = ""
    products_services: List[str] = Field(default_factory=list)
    hiring_signals: List[str] = Field(default_factory=list)
    website_url: str = ""


# Phone Status - explains why we do/don't have a phone
class PhoneStatus(str, Enum):
    FOUND_MOBILE = "found_mobile"           # Best case: mobile number found
    FOUND_LANDLINE = "found_landline"       # Landline found (less ideal)
    FILTERED_NON_DACH = "filtered_non_dach" # APIs returned phones but all non-German
    NO_LINKEDIN = "no_linkedin"             # Couldn't find LinkedIn profile
    NO_DECISION_MAKER = "no_decision_maker" # Couldn't identify a contact person
    API_NO_RESULT = "api_no_result"         # APIs returned nothing
    SKIPPED_PAID_API = "skipped_paid_api"   # Test mode - paid APIs skipped


# Final Enrichment Result
class EnrichmentResult(BaseModel):
    success: bool
    company: CompanyInfo
    company_intel: Optional[CompanyIntel] = None  # Sales research data
    decision_maker: Optional[DecisionMaker] = None
    phone: Optional[PhoneResult] = None
    phone_status: PhoneStatus = PhoneStatus.API_NO_RESULT  # Why we do/don't have a phone
    emails: List[str] = Field(default_factory=list)
    enrichment_path: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    # Warnings for n8n alerts (e.g. "primary_api_key_failed", "used_fallback_api_key")
    warnings: List[str] = Field(default_factory=list)

    # Operational alerts for n8n monitoring (extra fields, don't affect existing output)
    operational_alerts: Dict[str, bool] = Field(default_factory=dict)

    # Original input reference
    job_id: str
    job_title: str
