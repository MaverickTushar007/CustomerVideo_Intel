from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class RestaurantEvent(BaseModel):
    """
    Schema representing a single tracking, zone, or activity event inside the venue.
    """
    event_id: str = Field(..., description="Unique event UUID")
    camera_id: str = Field(..., description="Source camera identifier")
    track_id: str = Field(..., description="Unique tracker ID")
    zone_id: Optional[str] = Field(None, description="Semantic zone identifier")
    event_type: str = Field(..., description="Event type: e.g. enter_zone, exit_zone, activity_start, activity_end")
    start_ts: str = Field(..., description="ISO8601 UTC start timestamp")
    end_ts: Optional[str] = Field(None, description="ISO8601 UTC end timestamp")
    confidence: float = Field(..., description="Detector/tracker confidence score")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata including bounding boxes, lighting, occlusion, etc.")
