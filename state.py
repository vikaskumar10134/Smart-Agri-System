from pydantic import BaseModel , Field , field_validator
from typing import List , Literal , Optional , Annotated
from langgraph.graph.message import add_messages


class MemoryItem(BaseModel):

    text: str = Field(..., min_length=1, description="Atomic user memory, a single self-contained fact.")
    is_new: bool = Field(..., description="'True' if this fact is not already stored, 'False' if it duplicates an existing memory.")


class MemoryDecision(BaseModel):

    should_write: bool = Field(..., description="Whether any memories from this turn are worth persisting.")
    memories: List[MemoryItem] = Field(
        default_factory=list, description="Atomic candidate memories extracted from the turn."
    )


class FarmerProfile(BaseModel):
    location: dict = Field(
        ...,
        description=(
            "Farmer's field location. Expected keys: 'lat' (float), 'lon' (float), "
            "'state_name' (str), 'district' (str), 'market' (str, nearest APMC mandi)."
        ),
    )
    variety: str = Field(
        ...,
        description="Variety of the current crop (e.g. 'other', 'Pusa Purple Long' for Brinjal).",
    )
    grade: str = Field(
        ...,
        description="Market grade of the produce (e.g. 'Local', 'Grade-I', 'Grade-II').",
    )
    current_crop: Optional[str] = Field(
        default=None,
        description="Crop currently grown on the farmer's field, if known. None if unset.",
    )
    prefered_language: str = Field(
        ...,
        description="Farmer's preferred language for responses (e.g. 'English', 'Hindi').",
    )

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: dict) -> dict:
        required = {"lat", "lon"}
        missing = required - v.keys()
        if missing:
            raise ValueError(f"location is missing required keys: {missing}")
        if not (-90 <= v["lat"] <= 90):
            raise ValueError("location['lat'] must be between -90 and 90")
        if not (-180 <= v["lon"] <= 180):
            raise ValueError("location['lon'] must be between -180 and 180")
        return v

class WeatherForecast(BaseModel):
    source: str = Field(..., description="Name of the tool/service that produced this forecast, e.g. 'weather_mcp_server'.")
    fetch_at: str = Field(..., description="ISO-8601 UTC timestamp of when the forecast was fetched.")
    daily: List[dict] = Field(
        ...,
        description="Per-day forecast entries, each with keys 'date', 'rain_mm', 'temp_max_c', 'temp_min_c'.",
    )

    @field_validator("daily")
    @classmethod
    def validate_daily_not_empty(cls, v: List[dict]) -> List[dict]:
        if not v:
            raise ValueError("daily forecast list cannot be empty")
        return v

class MandiPricePoint(BaseModel):
    commodity: str = Field(..., description="Name of the crop/commodity, e.g. 'Brinjal'.")
    market: str = Field(..., description="Name of the mandi/market, e.g. 'Lalru APMC'.")
    date: str = Field(..., description="Date of the price observation, ISO-8601 (YYYY-MM-DD).")
    model_price_per_quintal: float = Field(
        ..., gt=0, description="Modal (most common) price per quintal in INR, must be positive."
    )
    source: str = Field(..., description="Name of the tool/API that supplied this price point.")

class PestDiagnosis(BaseModel):
    label: str = Field(..., description="Identified pest or disease name.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence for the label, between 0 and 1.")
    model_used: str = Field(..., description="Identifier of the vision model used, e.g. 'nvidia/vila'.")


class IntentSchema(BaseModel):
    intent: Literal["crop_selection", "irrigation_timing", "pest_diagnosis", "sell_or_hold_price"] = Field(
        ..., description="Single classified intent for the farmer's query."
    )


class DiagnosisSchema(BaseModel):
    label: str
    confidence: float
    action: str

    
class AgriAdvisoryState(BaseModel):
    message : Annotated[list , add_messages] = Field(
        default_factory=list, description="Running chat message history for this thread, merged via add_messages."
    )

    farmer_id: str = Field(..., description="Unique identifier for the farmer, used as the LTM store namespace key.")
    raw_query: str = Field(..., description="Farmer's original query text, in their own language.")
    detected_language: str = Field(..., description="Language code/name detected for raw_query, e.g. 'English', 'Hindi'.")
    query_type : str = Field(... , description='Type of the query whether it is text or image based')

    # router node output node
    intent: Optional[Literal["crop_selection", "irrigation_timing", "pest_diagnosis", "sell_or_hold_price"]] = Field(
        default=None, description="Classified intent for the query. None until intent_router_node runs."
    )

    intent_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Confidence score for the classified intent, between 0 and 1."
    )

    llm_used : int = Field(default=0 , description = 'LLM model used for intent classification, 0 for nvidia/llama-nemotron-13b-v2 , 1 for nvidia/llama-nemotron-13b-v2-chat')

    # from LTM
    farmer_profile: Optional[FarmerProfile] = Field(
        default=None, description="Farmer's long-term profile, loaded from PostgresStore by context_node."
    )

    # Weather mcp tool output
    weather_forcast: Optional[WeatherForecast] = Field(
        default=None, description="Weather forecast fetched via weather_mcp_server, if the intent required it."
    )
    weather_tool_error: Optional[str] = Field(
        default=None, description="Error message if the weather MCP tool call failed, else None."
    )

    # mandi price MCP tool
    mandi_prices: Optional[List[MandiPricePoint]] = Field(
        default=None, description="Recent mandi price points fetched via mandi_price_mcp_server, if required."
    )
    mandi_tool_error: Optional[str] = Field(
        default=None, description="Error message if the mandi price MCP tool call failed, else None."
    )

    # image branch(optional vision node)
    image_attach: bool = Field(
        default=False, description="Whether the farmer attached an image with this query."
    )
    image_url: Optional[str] = Field(
        default=None, description="URL/path of the attached crop image, required if image_attach is True."
    )
    pest_diagnosis: Optional[PestDiagnosis] = Field(
        default=None, description="Pest/disease diagnosis result from image_branch, if it ran."
    )

    # Confidence gate output
    data_freshness_ok: bool = Field(
        default=True, description="Whether all data backing the advisory is fresh/complete enough to present without caveats."
    )
    caveats: List[str] = Field(
        default_factory=list, description="Accumulated caveat tags (e.g. 'weather_unavailable') attached during the run."
    )

    # Advisiory node output
    advisory_draft: Optional[str] = Field(
        default=None, description="LLM-generated recommendation text before caveats/disclaimers are appended."
    )
    rag_snippet_used: Optional[List[str]] = Field(
        default=None, description="Agronomy knowledge-base snippets retrieved and used to ground the advisory."
    )

    # Respond node output
    final_response: Optional[str] = Field(
        default=None, description="Final farmer-facing response text, including any disclaimers."
    )
    tool_used: List[str] = Field(
        default_factory=list, description="Names of tools/models that contributed data to this turn's response."
    )

    @field_validator("image_url")
    @classmethod
    def validate_image_url_when_attached(cls, v, info):
        if info.data.get("image_attach") and not v:
            raise ValueError("image_url is required when image_attach is True")
        return v