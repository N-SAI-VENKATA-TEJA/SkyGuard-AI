from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict
import asyncio
import math
import pandas as pd
import numpy as np
import threading
import time

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

# Demo state
demo_running = False

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
    for k, v in result_dict.items():
        if isinstance(v, float) and math.isnan(v):
            result_dict[k] = None

    # Broadcast to websocket clients for this specific station
    await manager.broadcast_to_station(station_id, result_dict)
    
    return result_dict


def _run_demo_sync(station_id: str, loop):
    """Background thread: streams demo data with injected anomalies."""
    global demo_running
    import requests
    
    try:
        df = pd.read_csv('data/processed/aws_clean.csv', nrows=50)
        
        for i, row in df.iterrows():
            if not demo_running:
                break
                
            temp = float(row['temperature'])
            pres = float(row['pressure'])
            hum = float(row['humidity'])
            
            # Inject anomalies at specific points for demonstration
            if i == 38:  # Spike at row 38
                temp += 25.0
            elif i == 42:  # Frozen sensor at rows 42-46
                temp = float(df.iloc[42]['temperature'])
            elif 43 <= i <= 46:
                temp = float(df.iloc[42]['temperature'])
            elif i == 48:  # Data loss at row 48
                temp = None
                pres = None
                hum = None
                
            payload = {
                "station_id": station_id,
                "timestamp": row['timestamp'],
                "temperature": temp,
                "pressure": pres,
                "humidity": hum
            }
            
            try:
                requests.post("http://127.0.0.1:8000/api/v1/observations", json=payload, timeout=5)
            except Exception:
                pass
            
            time.sleep(1)
    finally:
        demo_running = False


@app.post("/api/v1/demo/start")
async def start_demo(station_id: str = "AWS_DEMO_01"):
    """Start a live demo that streams 50 observations with injected anomalies."""
    global demo_running
    
    if demo_running:
        return {"status": "already_running", "message": "Demo is already in progress."}
    
    # Reset station state for a fresh demo
    if station_id in pipelines:
        del pipelines[station_id]
    if station_id in last_timestamps:
        del last_timestamps[station_id]
    if station_id in station_locks:
        del station_locks[station_id]
    
    demo_running = True
    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=_run_demo_sync, args=(station_id, loop), daemon=True)
    thread.start()
    
    return {"status": "started", "message": f"Demo started for {station_id}. Watch the dashboard!"}


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

