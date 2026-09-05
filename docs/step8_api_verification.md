# Step 8.3: FastAPI + WebSocket API Verification

## 1. Health Test
- **Methodology**: `GET /health`
- **Result**: **PASS**. Returns 200 OK with `{"status": "ok"}`. ML pipelines are not instantiated.

## 2. Observation & Anomaly Tests
- **Methodology**: Posted sequentially to `/api/v1/observations` during the warmup phase, continuing into normal and anomalous rows.
- **Result**: **PASS**. Output correctly returns `WARMUP` for early observations, `PROCESSED` with `anomaly_flag=False` for normal rows, and transitions to `anomaly_flag=True` containing valid `fault_type` definitions when detecting spikes.

## 3. Station Isolation
- **Methodology**: Posted identical observations simultaneously to `AWS_001` and `AWS_002`. Verified distinct pipeline allocations.
- **Result**: **PASS**. `pipelines["AWS_001"]` resolves to an independent memory object from `pipelines["AWS_002"]`.

## 4. Ordering Test
- **Methodology**: Posted row at `12:10:00`, followed by a delayed row at `12:00:00`.
- **Result**: **PASS**. API responds with `400 Bad Request: Out-of-order observation. Latest processed: 2026-09-04 12:10:00`. The deque state is preserved safely.

## 5. WebSocket Delivery & Isolation
- **Methodology**: Connected WebSockets for `/ws/AWS_001` and `/ws/AWS_002`. Sent observation specifically to `AWS_001`.
- **Result**: **PASS**. `AWS_001` socket immediately receives the processed JSON JSON payload. `AWS_002` socket remains entirely silent.

## 6. State Continuity
- **Methodology**: Sent 40 sequential normal rows to a specific station identifier to clear warmup, followed by an anomalous row. 
- **Result**: **PASS**. The pipeline instance persisted. The sensor health transitioned strictly according to Step 7 continuity rules across independent HTTP requests (state did not reset per request).

## 7. API / StreamPipeline Equivalence
- **Methodology**: Passed 50 deterministic rows through the naked `StreamPipeline` object natively, and simultaneously sent the identical rows through `TestClient.post()`. 
- **Result**: **PASS**. Outputs match exactly, verifying the API layer induces zero alteration to ML processing logic.

## 8. Performance and Delivery Latency
- **Benchmark**: Evaluated end-to-end processing of a sequential 50-row batch via REST.
- **Mean API Latency**: 83.36 ms (includes HTTP JSON serialization + model inference overhead)
- **Median API Latency**: 51.08 ms
- **P95 API Latency**: 57.07 ms
- **Max API Latency**: 1640.58 ms (due to one-time lazy artifact cold-start/import on the first HTTP request).
- **Throughput**: ~12 HTTP POST requests per second (synchronously).
- **WebSocket Latency**: WebSocket delivery was empirically observed within the same HTTP request cycle during local testing. Browser end-to-end WebSocket latency was not independently measured. No "instantaneous" claim is made.

## Acceptance Criteria Checklist
- [x] FastAPI starts successfully
- [x] GET /health works
- [x] POST /api/v1/observations works
- [x] Existing StreamPipeline is reused
- [x] No frozen ML files are modified
- [x] Station-specific state is preserved
- [x] Out-of-order behavior is explicit
- [x] WebSocket endpoint works
- [x] Station-specific WebSocket broadcasting works
- [x] API result matches direct StreamPipeline result
- [x] Step 7 health state persists across API requests
- [x] Invalid requests are handled safely (422 and 400 errors)
- [x] /docs works (OpenAPI native)
- [x] Tests pass

**Verdict**: **PASS WITH LIMITATIONS**
