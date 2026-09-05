import pandas as pd
import requests

df = pd.read_csv("data/processed/aws_clean.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

for i, row in df.head(100).iterrows():
    payload = {
        "station_id": "AWS_LIVE_01",
        "timestamp": row["timestamp"].isoformat(),
        "temperature": float(row["temperature"]),
        "pressure": float(row["pressure"]),
        "humidity": float(row["humidity"])
    }

    response = requests.post(
        "http://127.0.0.1:8000/api/v1/observations",
        json=payload
    )

    data = response.json()

    print(
        f"[{i:4d}] "
        f"{data.get('processing_state', '?'):9s} | "
        f"flag={data.get('anomaly_flag')} | "
        f"{data.get('fault_type', '?'):30s} | "
        f"score={data.get('anomaly_score', 0):.2f}"
    )
