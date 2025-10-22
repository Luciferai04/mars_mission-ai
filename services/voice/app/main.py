#!/usr/bin/env python3
"""
Voice Command Interface Service

Accepts text commands (and optional audio via external STT) and maps to planning actions.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import httpx
import os

PLANNING_URL = os.getenv("PLANNING_SERVICE_URL", "http://planning-service:8005")

app = FastAPI(title="Voice Command Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CommandRequest(BaseModel):
    text: str
    context: Optional[Dict[str, Any]] = None


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "voice-service", "version": "1.0.0"}


@app.post("/command/execute")
async def execute_command(cmd: CommandRequest):
    text = cmd.text.lower().strip()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if text.startswith("plan mission"):
                # extract minimal params or use defaults
                ctx = cmd.context or {}
                payload = {
                    "sol": ctx.get("sol", 0),
                    "lat": ctx.get("lat", 18.44),
                    "lon": ctx.get("lon", 77.45),
                    "battery_soc": ctx.get("battery_soc", 0.6),
                    "time_budget_min": ctx.get("time_budget_min", 480),
                    "objectives": ctx.get("objectives", ["traverse", "image", "sample"]),
                }
                r = await client.post(f"{PLANNING_URL}/plan", json=payload)
                r.raise_for_status()
                return {"status": "ok", "action": "plan", "result": r.json()}
            elif text.startswith("status"):
                r = await client.get(f"{PLANNING_URL}/services/status")
                r.raise_for_status()
                return {"status": "ok", "action": "status", "result": r.json()}
            else:
                return {"status": "unknown_command"}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=503, detail=f"Planning service error: {e}")
