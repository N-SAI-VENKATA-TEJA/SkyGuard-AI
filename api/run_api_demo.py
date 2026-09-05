import time
import requests
import websocket
import threading
import json
import pandas as pd
import uvicorn
from contextlib import contextmanager

# Start server in background for demo purposes
def start_server():
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, log_level="error")

def run_demo():
    print("==================================================")
    print("SkyGuard AI - Real-Time API Demo")
    print("==================================================")
    
    station_id = "AWS_DEMO_01"
    
    # 1. Start WebSocket Listener
    ws = websocket.WebSocket()
    ws.connect(f"ws://127.0.0.1:8000/ws/{station_id}")
    print(f"[WS] Connected to ws://127.0.0.1:8000/ws/{station_id}")
    
    def listen_ws():
        while True:
            try:
                msg = ws.recv()
                data = json.loads(msg)
                flag = data.get('anomaly_flag', False)
                state = data.get('processing_state', 'UNKNOWN')
                print(f"[WS] Received result for {data['timestamp']}: State={state}, Anomaly={flag}")
            except:
                break
                
    ws_thread = threading.Thread(target=listen_ws, daemon=True)
    ws_thread.start()
    
    # 2. Performance Tracking
    latencies = []
    
    # Load sample data
    df = pd.read_csv('data/processed/aws_clean.csv', nrows=50)
    
    print("\n[HTTP] Sending sequential observations (1 per second)...")
    for i, row in df.iterrows():
        payload = {
            "station_id": station_id,
            "timestamp": row['timestamp'],
            "temperature": row['temperature'],
            "pressure": row['pressure'],
            "humidity": row['humidity']
        }
        
        t0 = time.time()
        resp = requests.post("http://127.0.0.1:8000/api/v1/observations", json=payload)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000)
        
        # Sleep for 1 second so you can watch it live on the dashboard!
        time.sleep(1) 
            
    # Allow WS to receive all
    time.sleep(1)
    
    # 3. Print Performance
    import numpy as np
    print("\n==================================================")
    print("API PERFORMANCE BENCHMARK (End-to-End HTTP Latency)")
    print(f"Mean Latency:   {np.mean(latencies):.2f} ms")
    print(f"Median Latency: {np.median(latencies):.2f} ms")
    print(f"P95 Latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"Max Latency:    {np.max(latencies):.2f} ms")
    print(f"Throughput:     {1000 / np.mean(latencies):.0f} req/sec")
    print("==================================================\n")

if __name__ == "__main__":
    run_demo()
