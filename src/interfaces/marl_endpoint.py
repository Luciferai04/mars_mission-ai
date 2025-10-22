#!/usr/bin/env python3
"""
MARL Optimization Endpoint

Add this to web_interface.py after the environment_search endpoint.
"""

from fastapi import HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


async def optimize_with_marl(mission_context: Dict[str, Any]):
    """Optimize mission plan using Multi-Agent Reinforcement Learning.
    
    Uses 5 specialized RL agents (route, power, science, hazard, strategy)
    trained on mission simulations to collaboratively optimize plans.
    
    Example payload:
    {
        "lat": 18.4447,
        "lon": 77.4508,
        "battery_soc": 0.65,
        "time_budget_min": 480,
        "targets": [
            {"id": "target_1", "lat": 18.45, "lon": 77.46, "priority": 10}
        ],
        "sol": 1600,
        "temp": -65,
        "dust": 0.45
    }
    """
    try:
        from src.core.multi_agent_rl import MultiAgentRLSystem
        
        # Initialize and load trained MARL system
        marl_system = MultiAgentRLSystem()
        try:
            marl_system.load_all_agents()
            logger.info("Loaded trained MARL agents")
        except Exception as load_error:
            logger.warning(f"Could not load trained agents: {load_error}. Using untrained agents.")
        
        # Optimize mission plan
        result = marl_system.optimize_mission_plan(mission_context)
        
        # Get training stats
        stats = marl_system.get_training_stats()
        
        return {
            "status": "success",
            "optimization": result,
            "marl_stats": {
                "trained_episodes": stats['episodes'],
                "avg_reward": stats['avg_reward'],
                "agent_confidence": {name: f"{(1-eps)*100:.1f}%" 
                                    for name, eps in stats['agent_epsilons'].items()}
            },
            "message": "Mission plan optimized using multi-agent reinforcement learning"
        }
        
    except Exception as e:
        logger.error(f"MARL optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"MARL error: {str(e)}")


# To add to web_interface.py, insert this line after other imports:
# from src.interfaces.marl_endpoint import optimize_with_marl

# Then add this route after the /environment_search endpoint:
# @app.post("/optimize_with_marl")
# async def marl_optimization(mission_context: Dict[str, Any]):
#     return await optimize_with_marl(mission_context)
