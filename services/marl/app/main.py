#!/usr/bin/env python3
"""
MARL Optimization Microservice
Multi-Agent Reinforcement Learning for mission optimization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path
import logging
import os

# Add parent directory to path to import MARL system
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.multi_agent_rl import MultiAgentRLSystem

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="MARL Optimization Service",
    description="Multi-Agent Reinforcement Learning for Mars mission optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global MARL system
marl_system = None


# Pydantic models
class MissionContext(BaseModel):
    lat: float
    lon: float
    battery_soc: float
    time_budget_min: int
    targets: List[Dict[str, Any]]
    sol: int
    temp: float
    dust: float


class OptimizationResponse(BaseModel):
    status: str
    optimized_actions: List[Dict[str, Any]]
    expected_completion: int
    total_power: float
    total_time: int
    rl_confidence: float
    agent_stats: Dict[str, str]


class AgentStats(BaseModel):
    episodes: int
    avg_reward: float
    agent_confidence: Dict[str, str]
    total_experiences: int


def load_marl_system():
    """Load MARL system and trained agents"""
    global marl_system
    
    try:
        model_dir = os.getenv('MODEL_DIR', '/models/marl')
        marl_system = MultiAgentRLSystem(models_dir=model_dir)
        
        # Try to load trained agents
        try:
            marl_system.load_all_agents()
            logger.info("Trained MARL agents loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load trained agents: {e}. Using untrained agents.")
        
    except Exception as e:
        logger.error(f"Error initializing MARL system: {e}")
        marl_system = None


@app.on_event("startup")
async def startup_event():
    """Load MARL system on startup"""
    logger.info("Starting MARL Optimization Service...")
    load_marl_system()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if marl_system is not None else "degraded",
        "service": "marl-service",
        "version": "1.0.0",
        "marl_loaded": marl_system is not None,
        "agents": 5 if marl_system else 0
    }


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_mission(context: MissionContext):
    """Optimize mission plan using MARL agents"""
    if marl_system is None:
        raise HTTPException(status_code=503, detail="MARL system not loaded")
    
    try:
        # Convert to dict for MARL system
        mission_context = context.dict()
        
        # Run optimization
        result = marl_system.optimize_mission_plan(mission_context)
        
        # Get stats
        stats = marl_system.get_training_stats()
        agent_confidence = {
            name: f"{(1-eps)*100:.1f}%" 
            for name, eps in stats['agent_epsilons'].items()
        }
        
        return OptimizationResponse(
            status="success",
            optimized_actions=result['optimized_actions'],
            expected_completion=result['expected_completion'],
            total_power=result['total_power'],
            total_time=result['total_time'],
            rl_confidence=result['rl_confidence'],
            agent_stats=agent_confidence
        )
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@app.get("/agents/stats", response_model=AgentStats)
async def get_agent_stats():
    """Get training statistics for all agents"""
    if marl_system is None:
        raise HTTPException(status_code=503, detail="MARL system not loaded")
    
    try:
        stats = marl_system.get_training_stats()
        
        return AgentStats(
            episodes=stats['episodes'],
            avg_reward=stats['avg_reward'],
            agent_confidence={
                name: f"{(1-eps)*100:.1f}%" 
                for name, eps in stats['agent_epsilons'].items()
            },
            total_experiences=stats['total_experiences']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@app.get("/agents/confidence")
async def get_agent_confidence():
    """Get confidence levels for all agents"""
    if marl_system is None:
        raise HTTPException(status_code=503, detail="MARL system not loaded")
    
    stats = marl_system.get_training_stats()
    
    return {
        "confidence": {
            name: {
                "exploitation_rate": f"{(1-eps)*100:.1f}%",
                "exploration_rate": f"{eps*100:.1f}%",
                "epsilon": eps
            }
            for name, eps in stats['agent_epsilons'].items()
        }
    }


@app.post("/agents/reload")
async def reload_agents():
    """Reload trained agents (hot-swap)"""
    try:
        if marl_system is None:
            load_marl_system()
        else:
            marl_system.load_all_agents()
        
        return {
            "status": "success",
            "agents_loaded": marl_system is not None,
            "message": "Agents reloaded successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")


@app.post("/train")
async def train_agents(episodes: int = 100):
    """Train MARL agents (optional, for continuous learning)"""
    if marl_system is None:
        raise HTTPException(status_code=503, detail="MARL system not loaded")
    
    # Note: Training is computationally expensive
    # In production, this should be done offline or in a separate worker
    return {
        "status": "not_implemented",
        "message": "Training should be done offline. Use scripts/train_marl.py",
        "recommended_command": f"python scripts/train_marl.py --episodes {episodes}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
