#!/usr/bin/env python3
"""
Training script for Multi-Agent Reinforcement Learning System

Trains 5 specialized RL agents to optimize Mars mission planning through
collaborative decision-making and continuous learning from mission outcomes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.multi_agent_rl import MultiAgentRLSystem, AgentState, train_marl_system
import numpy as np
import logging
import argparse
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train MARL system for Mars mission planning')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save checkpoint every N episodes (default: 100)')
    parser.add_argument('--test', action='store_true',
                       help='Test existing trained model')
    args = parser.parse_args()
    
    if args.test:
        logger.info("Loading and testing existing MARL system...")
        system = MultiAgentRLSystem()
        system.load_all_agents()
        
        # Test with sample mission context
        test_context = {
            'lat': 18.4447,
            'lon': 77.4508,
            'battery_soc': 0.65,
            'time_budget_min': 480,
            'targets': [
                {'id': 'delta_sample_1', 'lat': 18.45, 'lon': 77.46, 'priority': 10},
                {'id': 'rock_outcrop_2', 'lat': 18.46, 'lon': 77.47, 'priority': 8},
                {'id': 'crater_floor_3', 'lat': 18.44, 'lon': 77.45, 'priority': 6}
            ],
            'sol': 1600,
            'temp': -65,
            'dust': 0.45
        }
        
        result = system.optimize_mission_plan(test_context)
        
        print("\n" + "="*60)
        print("MARL-OPTIMIZED MISSION PLAN")
        print("="*60)
        print(f"\nOptimized Actions: {len(result['optimized_actions'])}")
        print(f"Expected Completions: {result['expected_completion']}")
        print(f"Total Power: {result['total_power']:.1f} Wh")
        print(f"Total Time: {result['total_time']} min")
        print(f"RL Confidence: {result['rl_confidence']:.2%}")
        
        print("\nAction Sequence:")
        for i, action in enumerate(result['optimized_actions'][:5], 1):
            print(f"  {i}. {action['type'].upper()}: "
                  f"Target={action['target']}, "
                  f"Duration={action['duration']}min, "
                  f"Power={action['power']:.1f}W")
        
        stats = system.get_training_stats()
        print(f"\nTraining History:")
        print(f"  Episodes: {stats['episodes']}")
        print(f"  Average Reward: {stats['avg_reward']:.2f}")
        print(f"  Agent Exploration Rates:")
        for agent_name, epsilon in stats['agent_epsilons'].items():
            print(f"    {agent_name}: {epsilon:.3f} (exploitation: {(1-epsilon)*100:.1f}%)")
        
    else:
        logger.info(f"Starting MARL training for {args.episodes} episodes...")
        start_time = datetime.now()
        
        system = train_marl_system(
            episodes=args.episodes,
            save_interval=args.save_interval
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        stats = system.get_training_stats()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nTraining Duration: {duration/60:.1f} minutes")
        print(f"Total Episodes: {stats['episodes']}")
        print(f"Average Reward (last 100): {stats['avg_reward']:.2f}")
        print(f"Total Experiences Collected: {stats['total_experiences']}")
        
        print("\nFinal Agent States:")
        for agent_name, epsilon in stats['agent_epsilons'].items():
            exploitation = (1 - epsilon) * 100
            print(f"  {agent_name.capitalize()}: "
                  f"ε={epsilon:.3f} (exploitation {exploitation:.1f}%)")
        
        print(f"\nModels saved to: models/marl/")
        print("\nTo test the trained system, run:")
        print(f"  python {sys.argv[0]} --test")


if __name__ == "__main__":
    main()
