from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import asyncio
import pandas as pd

from api.schemas import ObservationRequest, ObservationResponse
from api.connection_manager import manager
from ml.anomaly_engine.stream_pipeline import StreamPipeline

app = FastAPI(
    title="SkyGuard AI Real-Time API",
    description="Real-time fault classification and sensor health streaming API.",
    version="1.0.0"
)

# Allow CORS for prototype dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Station State Management
pipelines: Dict[str, StreamPipeline] = {}
station_locks: Dict[str, asyncio.Lock] = {}
last_timestamps: Dict[str, pd.Timestamp] = {}

def get_station_lock(station_id: str) -> asyncio.Lock:
    if station_id not in station_locks:
        station_locks[station_id] = asyncio.Lock()
    return station_locks[station_id]

def get_pipeline(station_id: str) -> StreamPipeline:
    if station_id not in pipelines:
        pipelines[station_id] = StreamPipeline()
    return pipelines[station_id]

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "SkyGuard AI"
    }

@app.post("/api/v1/observations", response_model=ObservationResponse)
async def process_observation(obs: ObservationRequest):
    station_id = obs.station_id
    lock = get_station_lock(station_id)
    
    # Convert incoming timestamp to pandas Timestamp
    try:
        current_ts = pd.Timestamp(obs.timestamp)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    async with lock:
        # Check ordering
        if station_id in last_timestamps:
            if current_ts <= last_timestamps[station_id]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Out-of-order observation. Latest processed: {last_timestamps[station_id]}"
                )
                
        pipeline = get_pipeline(station_id)
        
        try:
            # Process single observation synchronously
            result_dict = pipeline.process_observation(
                timestamp=current_ts,
                temperature=obs.temperature if obs.temperature is not None else float('nan'),
                pressure=obs.pressure if obs.pressure is not None else float('nan'),
                humidity=obs.humidity if obs.humidity is not None else float('nan')
            )
        except Exception as e:
            # Catch internal pipeline errors without exposing raw stack trace
            raise HTTPException(status_code=500, detail="Internal processing error")

        # Update last processed timestamp
        last_timestamps[station_id] = current_ts

    # Add station_id to result for broadcasting contract
    result_dict["station_id"] = station_id
    
    # Convert timestamp back to ISO string for JSON
    result_dict["timestamp"] = current_ts.isoformat()
    
    # Sanitize NaNs to None for strict JSON compliance in Starlette
    import math
    for k, v in result_dict.items():
        if isinstance(v, float) and math.isnan(v):
            result_dict[k] = None

    # Broadcast to websocket clients for this specific station
    await manager.broadcast_to_station(station_id, result_dict)
    
    return result_dict

@app.websocket("/ws/{station_id}")
async def websocket_endpoint(websocket: WebSocket, station_id: str):
    await manager.connect(websocket, station_id)
    try:
        while True:
            # Keep connection alive; clients generally just listen,
            # but we can receive ping/pong or dummy messages here.
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, station_id)
