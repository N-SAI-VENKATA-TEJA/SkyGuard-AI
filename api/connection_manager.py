from fastapi import WebSocket
from typing import Dict, List
import json

class ConnectionManager:
    def __init__(self):
        # Maps station_id to a list of connected WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, station_id: str):
        await websocket.accept()
        if station_id not in self.active_connections:
            self.active_connections[station_id] = []
        self.active_connections[station_id].append(websocket)

    def disconnect(self, websocket: WebSocket, station_id: str):
        if station_id in self.active_connections:
            if websocket in self.active_connections[station_id]:
                self.active_connections[station_id].remove(websocket)
            if not self.active_connections[station_id]:
                del self.active_connections[station_id]

    async def broadcast_to_station(self, station_id: str, message: dict):
        if station_id in self.active_connections:
            # We create a copy of the list to safely iterate while items might be removed on error
            for connection in list(self.active_connections[station_id]):
                try:
                    await connection.send_text(json.dumps(message))
                except Exception:
                    self.disconnect(connection, station_id)

manager = ConnectionManager()
