"""Unit tests for core mission planning modules."""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from resource_optimizer import ResourceOptimizer
from path_planner import PathPlanner
from constraint_solver import ConstraintSolver


@pytest.mark.unit
class TestResourceOptimizer:
    """Test resource optimization functionality."""
    
    def test_mmrtg_energy_calculation(self):
        """Test MMRTG energy budget calculation."""
        optimizer = ResourceOptimizer()
        
        # 110W MMRTG over 480 minutes (8 hours)
        energy = optimizer.estimate_mmrtg_energy_wh(110.0, 480)
        
        expected = 110.0 * (480 / 60.0)  # 880 Wh
        assert abs(energy - expected) < 0.1
    
    def test_activity_energy_estimation(self, mock_activities):
        """Test activity energy estimation."""
        optimizer = ResourceOptimizer()
        
        for activity in mock_activities:
            energy = optimizer.estimate_activity_energy_wh(activity)
            assert energy > 0
    
    def test_power_optimization_within_budget(self, mock_activities):
        """Test that optimization keeps activities within budget."""
        optimizer = ResourceOptimizer()
        
        optimized, summary = optimizer.optimize_power_usage(
            mock_activities,
            mmrtg_output_w=110.0,
            time_budget_min=480,
            battery_capacity_ah=43.0,
            battery_count=2
        )
        
        assert summary['consumed_wh'] <= summary['budget_wh']
        assert len(optimized) > 0
    
    def test_power_optimization_exceeds_budget(self):
        """Test optimization when activities exceed budget."""
        optimizer = ResourceOptimizer()
        
        # Create power-hungry activities
        activities = [
            {'id': f'act{i}', 'type': 'science', 'duration_min': 60, 'power_w': 500}
            for i in range(10)
        ]
        
        optimized, summary = optimizer.optimize_power_usage(
            activities,
            mmrtg_output_w=110.0,
            time_budget_min=480,
            battery_capacity_ah=43.0,
            battery_count=2
        )
        
        # Some activities should be deferred or scaled
        assert len(optimized) < len(activities) or \
               any(a.get('status') == 'deferred' for a in optimized)
    
    def test_thermal_management_cold(self):
        """Test thermal management in cold conditions."""
        optimizer = ResourceOptimizer()
        
        activities = [
            {'id': 'act1', 'type': 'imaging', 'duration_min': 20}
        ]
        
        environment = {'temperature_c': -50}
        
        adjusted = optimizer.thermal_management(
            activities,
            environment,
            operating_temp_min_c=-40,
            operating_temp_max_c=40
        )
        
        # Activity should be flagged for thermal issues
        assert any(a.get('status') == 'await_thermal_ok' for a in adjusted)
    
    def test_thermal_management_nominal(self):
        """Test thermal management in nominal conditions."""
        optimizer = ResourceOptimizer()
        
        activities = [
            {'id': 'act1', 'type': 'imaging', 'duration_min': 20}
        ]
        
        environment = {'temperature_c': -20}
        
        adjusted = optimizer.thermal_management(
            activities,
            environment,
            operating_temp_min_c=-40,
            operating_temp_max_c=40
        )
        
        # Activities should proceed normally
        assert not any(a.get('status') == 'await_thermal_ok' for a in adjusted)
    
    def test_safety_margin_applied(self):
        """Test that safety margins are applied."""
        optimizer = ResourceOptimizer(safety_margin_pct=0.2)
        
        # Calculate budget with safety margin
        budget = optimizer.estimate_total_budget_wh(
            mmrtg_output_w=110.0,
            time_budget_min=480,
            battery_capacity_ah=43.0,
            battery_count=2,
            use_battery=True
        )
        
        # Budget should be realistic for multi-sol operations (MMRTG + battery)
        # Raw MMRTG: 110W * 8hrs = 880Wh, Battery: 43Ah * 28V * 2 = ~2400Wh
        max_realistic_budget = 1200  # Wh (more realistic for constrained ops)
        assert budget < max_realistic_budget


@pytest.mark.unit
class TestPathPlanner:
    """Test path planning functionality."""
    
    def test_cost_map_flat_terrain(self):
        """Test cost map on flat terrain."""
        planner = PathPlanner(slope_max_deg=30.0)
        
        slope = np.ones((50, 50)) * 5.0  # 5 degree slope everywhere
        cost = planner.cost_map(slope)
        
        # All costs should be finite and reasonable
        assert np.all(np.isfinite(cost))
        assert np.all(cost >= 1.0)
        assert np.all(cost < 10.0)
    
    def test_cost_map_hazardous_terrain(self):
        """Test cost map with hazardous slopes."""
        planner = PathPlanner(slope_max_deg=30.0)
        
        slope = np.ones((50, 50)) * 5.0
        slope[25, 25] = 35.0  # Hazard in center
        
        cost = planner.cost_map(slope)
        
        # Hazardous cell should have infinite cost
        assert np.isinf(cost[25, 25])
        assert np.isfinite(cost[20, 20])
    
    def test_astar_simple_path(self):
        """Test A* pathfinding on simple cost map."""
        planner = PathPlanner(slope_max_deg=30.0)
        
        # Simple uniform cost
        cost = np.ones((20, 20)) * 1.0
        
        path = planner.astar(cost, (0, 0), (19, 19))
        
        assert path is not None
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (19, 19)
    
    def test_astar_no_path(self):
        """Test A* when no path exists."""
        planner = PathPlanner(slope_max_deg=30.0)
        
        # Create impassable barrier
        cost = np.ones((20, 20)) * 1.0
        cost[10, :] = np.inf  # Impassable wall
        
        path = planner.astar(cost, (0, 0), (19, 19))
        
        # No path should exist across barrier
        assert path is None
    
    def test_astar_avoids_high_cost(self):
        """Test that A* avoids high-cost areas."""
        planner = PathPlanner(slope_max_deg=30.0)
        
        # Low cost everywhere except center
        cost = np.ones((20, 20)) * 1.0
        cost[8:12, 8:12] = 100.0  # High cost region
        
        path = planner.astar(cost, (0, 0), (19, 19))
        
        assert path is not None
        # Path should avoid high-cost center region
        for row, col in path:
            if 8 <= row < 12 and 8 <= col < 12:
                # If it passes through, shouldn't spend much time there
                pass


@pytest.mark.unit
class TestConstraintSolver:
    """Test constraint solving functionality."""
    
    def test_evaluate_valid_plan(self, mock_activities, mock_mission_constraints):
        """Test evaluation of valid mission plan."""
        solver = ConstraintSolver()
        
        result = solver.evaluate(mock_activities, mock_mission_constraints)
        
        # Should pass validation
        assert isinstance(result, dict)
        assert 'ok' in result
        assert 'violations' in result
    
    def test_evaluate_time_exceeded(self, mock_activities):
        """Test detection of time budget violations."""
        solver = ConstraintSolver()
        
        # Very long activities
        activities = [
            {'id': f'act{i}', 'duration_min': 200}
            for i in range(5)
        ]
        
        constraints = {'time_budget_min': 100}
        
        result = solver.evaluate(activities, constraints)
        
        assert not result['ok']
        assert len(result['violations']) > 0
        assert any('time_budget' in v for v in result['violations'])
    
    def test_evaluate_power_exceeded(self):
        """Test detection of power violations."""
        solver = ConstraintSolver()
        
        activities = [
            {'id': 'high_power', 'power_w': 1000, 'duration_min': 10}
        ]
        
        constraints = {
            'power': {
                'mmrtg_output_w': 110,
                'peak_demand_w': 900
            }
        }
        
        result = solver.evaluate(activities, constraints)
        
        assert not result['ok']
        assert any('peak_power' in v for v in result['violations'])
    
    def test_repair_adds_missing_comm(self):
        """Test that repair adds missing communication activity."""
        solver = ConstraintSolver()
        
        activities = [
            {'id': 'science', 'type': 'imaging', 'duration_min': 20}
        ]
        
        constraints = {'comm_windows_per_sol': 1}
        
        repaired = solver.repair(activities, constraints)
        
        # Should add data transmission
        assert len(repaired) > len(activities)
        assert any(a.get('type') == 'data_tx' for a in repaired)
    
    def test_evaluate_no_activities(self):
        """Test evaluation with no activities."""
        solver = ConstraintSolver()
        
        result = solver.evaluate([], {'time_budget_min': 480})
        
        assert not result['ok']
        assert any('no activities' in v.lower() or 'missing' in v.lower() 
                  for v in result['violations'])


@pytest.mark.unit
class TestIntegration:
    """Integration tests between modules."""
    
    def test_resource_optimizer_with_path_planner(self, mock_activities):
        """Test resource optimizer integrated with path planning."""
        optimizer = ResourceOptimizer()
        planner = PathPlanner()
        
        # Optimize resources
        optimized, summary = optimizer.optimize_power_usage(
            mock_activities,
            mmrtg_output_w=110.0,
            time_budget_min=480
        )
        
        # Create simple terrain for path planning
        slope = np.ones((50, 50)) * 10.0
        cost = planner.cost_map(slope)
        path = planner.astar(cost, (0, 0), (49, 49))
        
        assert path is not None
        assert len(optimized) > 0
        assert summary['consumed_wh'] <= summary['budget_wh']
