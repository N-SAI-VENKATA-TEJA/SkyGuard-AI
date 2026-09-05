from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict

class ShapFeature(BaseModel):
    feature: str
    contribution: float

class ObservationRequest(BaseModel):
    station_id: str = Field(..., description="Unique identifier for the AWS station")
    timestamp: str = Field(..., description="ISO 8601 timestamp of the observation")
    temperature: Optional[float] = Field(None, description="Temperature reading")
    pressure: Optional[float] = Field(None, description="Pressure reading")
    humidity: Optional[float] = Field(None, description="Humidity reading")

class ObservationResponse(BaseModel):
    station_id: str
    timestamp: str
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[float] = None
    
    processing_state: str
    
    anomaly_score: float
    anomaly_flag: bool
    severity: str
    confidence: float
    
    fault_type: str
    affected_sensor: str
    
    sensor_health_temperature: float
    sensor_health_pressure: float
    sensor_health_humidity: float
    
    temperature_status: str
    pressure_status: str
    humidity_status: str
    
    data_quality_status: str
    maintenance_status: str
    explanation: str
    
    # Explainable AI: SHAP top contributing features
    shap_top_features: Optional[List[ShapFeature]] = None
    
    # Corrected/imputed value suggestions
    suggested_corrections: Optional[Dict[str, float]] = None
