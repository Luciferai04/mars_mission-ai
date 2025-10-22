#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Dict
import numpy as np
import os

app = FastAPI(title="3D Terrain Visualization Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/heightmap")
async def get_heightmap(npy_path: str = "data/dem/jezero_demo.npy", downsample: int = 4):
    try:
        arr = np.load(npy_path)
        arr = arr[::downsample, ::downsample]
        # Normalize to 0..1
        mn, mx = float(arr.min()), float(arr.max())
        norm = (arr - mn) / (mx - mn + 1e-9)
        return {
            "width": int(norm.shape[1]),
            "height": int(norm.shape[0]),
            "data": norm.astype(float).tolist(),
            "min": mn,
            "max": mx,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
