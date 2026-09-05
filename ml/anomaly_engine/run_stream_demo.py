import argparse
import pandas as pd
import time
from ml.anomaly_engine.stream_pipeline import StreamPipeline

def run_demo(rows):
    print("==================================================")
    print("STEP 8.2: STREAM PIPELINE DEMONSTRATION")
    print("==================================================")
    
    pipeline = StreamPipeline()
    
    # Load dataset to simulate stream
    df = pd.read_csv('data/processed/aws_clean.csv', nrows=rows)
    
    print(f"Streaming {rows} observations...")
    print(f"{'Timestamp':<22} | {'State':<9} | {'Anomaly':<7} | {'Type':<12} | {'Sensor':<10} | {'T_Health':<8} | {'Latency(ms)':<10}")
    print("-" * 95)
    
    latencies = []
    
    for _, row in df.iterrows():
        t0 = time.perf_counter()
        
        res = pipeline.process_observation(
            timestamp=row['timestamp'],
            temperature=row['temperature'],
            pressure=row['pressure'],
            humidity=row['humidity']
        )
        
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        latencies.append(lat_ms)
        
        flag_str = "YES" if res['anomaly_flag'] else "NO"
        
        print(f"{res['timestamp']:<22} | {res['processing_state']:<9} | {flag_str:<7} | "
              f"{res['fault_type']:<12} | {res['affected_sensor']:<10} | "
              f"{res['sensor_health_temperature']:<8.1f} | {lat_ms:<10.2f}")
              
    print("==================================================")
    print("PERFORMANCE BENCHMARK")
    import numpy as np
    print(f"Mean Latency:   {np.mean(latencies):.2f} ms")
    print(f"Median Latency: {np.median(latencies):.2f} ms")
    print(f"P95 Latency:    {np.percentile(latencies, 95):.2f} ms")
    print(f"Max Latency:    {np.max(latencies):.2f} ms")
    print(f"Throughput:     {1000.0 / np.mean(latencies):.0f} rows/sec")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=100)
    args = parser.parse_args()
    run_demo(args.rows)
